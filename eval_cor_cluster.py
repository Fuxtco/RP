# Correlation clustering evaluation

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from src.cluster_viz import save_two_row_figure
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from src.utils import (
    bool_flag,
    fix_random_seeds,
    init_distributed_mode,
    initialize_exp,
    # restart_from_checkpoint,
)

import src.resnet50 as resnet_models

logger = getLogger()

def summarize_features(name: str, feats: np.ndarray, logger_):
    """
    feats: (N, D) float32/float64
    打印：
      - std：mean/min/max
      - off-diagonal cosine similarity 的 min/p1/p50/p99/max
    """
    if feats is None or feats.size == 0:
        logger_.info(f"[{name}] empty")
        return

    x = feats.astype(np.float32)
    # std over samples, per-dim
    std = x.std(axis=0)
    logger_.info(
        f"[{name}] std_mean={std.mean():.6f} std_min={std.min():.6f} std_max={std.max():.6f}"
    )

    # cosine sim over samples (need L2)
    x = l2_normalize(x)
    N = x.shape[0]
    if N <= 2000:
        sim = x @ x.T
        off = sim[~np.eye(N, dtype=bool)]
        p1, p50, p99 = np.percentile(off, [1, 50, 99])
        logger_.info(
            f"[{name}] cos(offdiag) min={off.min():.6f} p1={p1:.6f} p50={p50:.6f} p99={p99:.6f} max={off.max():.6f}"
        )
    else:
        rng = np.random.RandomState(0)
        idx = rng.choice(N, size=min(512, N), replace=False)
        xs = x[idx]
        sim = xs @ xs.T
        off = sim[~np.eye(xs.shape[0], dtype=bool)]
        p1, p50, p99 = np.percentile(off, [1, 50, 99])
        logger_.info(
            f"[{name}] cos(offdiag, sampled) min={off.min():.6f} p1={p1:.6f} p50={p50:.6f} p99={p99:.6f} max={off.max():.6f}"
        )

# ---------------------------
# Metrics / helpers
# ---------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # purity = sum_k max_j |C_k ∩ L_j| / N
    N = len(y_true)
    if N == 0:
        return 0.0
    purity = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0:
            continue
        true_labels = y_true[idx]
        counts = np.bincount(true_labels)
        purity += int(counts.max()) if counts.size > 0 else 0
    return float(purity) / float(N)

# ---------------------------
# Pairwise + clustering metrics (paper-aligned)
# ---------------------------
def rand_index_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Rand Index (RI), not adjusted.
    RI = (TP + TN) / (total_pairs)
    """
    n = len(y_true)
    if n <= 1:
        return 1.0
    # pairwise same/diff matrices
    same_true = (y_true[:, None] == y_true[None, :])
    same_pred = (y_pred[:, None] == y_pred[None, :])
    # consider upper triangle excluding diagonal
    iu = np.triu_indices(n, k=1)
    st = same_true[iu]
    sp = same_pred[iu]
    tp = np.logical_and(st, sp).sum()
    tn = np.logical_and(~st, ~sp).sum()
    total = len(st)
    return float(tp + tn) / float(total)

def variation_of_information(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    VI(U,V) = H(U) + H(V) - 2 I(U;V)
    Using natural log (base e). Matches typical definition up to constant factor.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    # relabel to 0..K-1
    _, yt = np.unique(y_true, return_inverse=True)
    _, yp = np.unique(y_pred, return_inverse=True)

    Kt = yt.max() + 1
    Kp = yp.max() + 1

    # contingency
    cont = np.zeros((Kt, Kp), dtype=np.float64)
    for i in range(n):
        cont[yt[i], yp[i]] += 1.0
    cont /= float(n)

    pt = cont.sum(axis=1, keepdims=True)  # (Kt,1)
    pp = cont.sum(axis=0, keepdims=True)  # (1,Kp)

    # entropies
    Ht = -np.sum(pt * np.log(pt + eps))
    Hp = -np.sum(pp * np.log(pp + eps))

    # mutual information
    MI = np.sum(cont * (np.log(cont + eps) - np.log(pt + eps) - np.log(pp + eps)))
    VI = float(Ht + Hp - 2.0 * MI)
    return VI

def pairwise_metrics_from_similarity(sim: np.ndarray, y_true: np.ndarray, m: float) -> Dict[str, float]:
    """
    Evaluate independent pair classification using similarity threshold.
    Predict "join" (same cluster) if sim >= m.
    Cuts = different cluster.
    Returns ACC, PC, RC, PJ, RJ, F1C, F1J.
    """
    n = len(y_true)
    if n <= 1:
        return {"ACC": 1.0, "PC": 1.0, "RC": 1.0, "PJ": 1.0, "RJ": 1.0, "F1C": 1.0, "F1J": 1.0}

    iu = np.triu_indices(n, k=1)
    sim_u = sim[iu]
    pred_join = (sim_u >= m)

    true_join = (y_true[iu[0]] == y_true[iu[1]])
    true_cut = ~true_join
    pred_cut = ~pred_join

    # ACC
    acc = float((pred_join == true_join).sum()) / float(len(true_join))

    # joins precision/recall
    tpJ = np.logical_and(pred_join, true_join).sum()
    fpJ = np.logical_and(pred_join, true_cut).sum()
    fnJ = np.logical_and(pred_cut, true_join).sum()

    PJ = float(tpJ) / float(tpJ + fpJ + 1e-12)
    RJ = float(tpJ) / float(tpJ + fnJ + 1e-12)
    F1J = 2.0 * PJ * RJ / (PJ + RJ + 1e-12)

    # cuts precision/recall
    tpC = np.logical_and(pred_cut, true_cut).sum()
    fpC = np.logical_and(pred_cut, true_join).sum()
    fnC = np.logical_and(pred_join, true_cut).sum()

    PC = float(tpC) / float(tpC + fpC + 1e-12)
    RC = float(tpC) / float(tpC + fnC + 1e-12)
    F1C = 2.0 * PC * RC / (PC + RC + 1e-12)

    return {"ACC": acc, "PC": PC, "RC": RC, "PJ": PJ, "RJ": RJ, "F1C": F1C, "F1J": F1J}

def pairwise_metrics_from_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    logit_thresh: float = 0.0,
) -> Dict[str, float]:
    """
    logits: (N,N) symmetric, diagonal ignored
    y_true: (N,)
    join iff logit > logit_thresh
    Returns ACC, PC, RC, PJ, RJ, F1C, F1J (same keys as cosine pipeline)
    """
    n = len(y_true)
    if n <= 1:
        return {"ACC": 1.0, "PC": 1.0, "RC": 1.0, "PJ": 1.0, "RJ": 1.0, "F1C": 1.0, "F1J": 1.0}

    iu = np.triu_indices(n, k=1)
    log_u = logits[iu]
    pred_join = (log_u > float(logit_thresh))

    true_join = (y_true[iu[0]] == y_true[iu[1]])
    true_cut = ~true_join
    pred_cut = ~pred_join

    # ACC
    acc = float((pred_join == true_join).sum()) / float(len(true_join))

    # joins precision/recall
    tpJ = np.logical_and(pred_join, true_join).sum()
    fpJ = np.logical_and(pred_join, true_cut).sum()
    fnJ = np.logical_and(pred_cut, true_join).sum()

    PJ = float(tpJ) / float(tpJ + fpJ + 1e-12)
    RJ = float(tpJ) / float(tpJ + fnJ + 1e-12)
    F1J = 2.0 * PJ * RJ / (PJ + RJ + 1e-12)

    # cuts precision/recall
    tpC = np.logical_and(pred_cut, true_cut).sum()
    fpC = np.logical_and(pred_cut, true_join).sum()
    fnC = np.logical_and(pred_join, true_cut).sum()

    PC = float(tpC) / float(tpC + fpC + 1e-12)
    RC = float(tpC) / float(tpC + fnC + 1e-12)
    F1C = 2.0 * PC * RC / (PC + RC + 1e-12)

    return {"ACC": acc, "PC": PC, "RC": RC, "PJ": PJ, "RJ": RJ, "F1C": F1C, "F1J": F1J}

def cosine_similarity_matrix(feats_l2: np.ndarray) -> np.ndarray:
    """
    feats_l2 must be L2-normalized. Then cosine sim = dot.
    """
    return feats_l2 @ feats_l2.T

def save_cosine_hist(sim: np.ndarray, out_path: str, title: str, bins: int = 80):
    """
    Save histogram of pairwise cosine similarities (upper triangle, excluding diagonal).
    """
    n = sim.shape[0]
    if n <= 1:
        return
    iu = np.triu_indices(n, k=1)
    vals = sim[iu].astype(np.float64)

    plt.figure()
    plt.hist(vals, bins=bins)
    plt.title(title)
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_m_sweep_plot(m_rows: List[Dict[str, float]], metric: str, out_path: str, title: str):
    """
    Save line plot: metric vs m from sweep CSV rows.
    """
    if not m_rows:
        return
    ms = [float(r["m"]) for r in m_rows if "m" in r]
    ys = [float(r.get(metric, float("nan"))) for r in m_rows]

    plt.figure()
    plt.plot(ms, ys)
    plt.title(title)
    plt.xlabel("m")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def cosine_margin_weights(sim: np.ndarray, m: float) -> np.ndarray:
    """
    Build ILP weights w_pq from cosine and margin m.
    Your ILP decoder expects:
      w_pq > 0  => prefer JOIN
      w_pq < 0  => prefer CUT

    With cost c_pq = m - cos:
      cos > m  => prefer JOIN
      cos < m  => prefer CUT

    So choose w = cos - m.
    """
    w = sim.astype(np.float32) - float(m)
    np.fill_diagonal(w, 0.0)
    return w

def is_image_file(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


# ---------------------------
# Datasets (labeled & unlabeled)
# ---------------------------
class UnlabeledFolderDataset(Dataset):
    """
    Read images from a folder recursively WITHOUT requiring class subfolders.
    Returns (image_tensor, dummy_label, path).
    """
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.paths: List[str] = []
        for r, _, files in os.walk(root):
            for fn in files:
                p = os.path.join(r, fn)
                if is_image_file(p):
                    self.paths.append(p)
        self.paths.sort()
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0, path


class LabeledImageFolderWithPath(datasets.ImageFolder):
    """
    ImageFolder but returns (image_tensor, label, path).
    No special 16-bit handling; images are loaded via PIL and converted to RGB.
    """
    def __init__(self, root: str, transform=None):
        super().__init__(root=root, transform=transform)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = self.loader(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target, path


def build_eval_transform(img_size: int = 224) -> transforms.Compose:
    # keep consistent with MultiCropDataset normalization values
    # mean = [0.485, 0.456, 0.406]
    mean=[0.0832, 0.0897, 0.0733]
    # std = [0.228, 0.224, 0.225]
    std=[0.0976, 0.1375, 0.0852]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_dataset(split_path: str, img_size: int, unlabeled: bool) -> Dataset:
    t = build_eval_transform(img_size)
    if unlabeled:
        return UnlabeledFolderDataset(split_path, transform=t)
    return LabeledImageFolderWithPath(split_path, transform=t)


# ---------------------------
# Embedding extraction
# ---------------------------
@dataclass
class EmbedResult:
    feats: np.ndarray   #2048/128
    y_true: Optional[np.ndarray]
    paths: List[str]


class IndexedDataset(Dataset):
    """
    Wrap any dataset so that it also returns the global index `idx`.
    Works with ConcatDataset because idx here is the *global* index of the ConcatDataset.
    """
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y, p = self.base[idx]   # expect (img, label, path)
        return x, y, p, idx        # attach global idx

@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    workers: int,
    device: str,
    distributed: bool,
    use_proj: bool,
    feat_dim: int,
) -> EmbedResult:

    if distributed and dist.is_available() and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=False, drop_last=False
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    if sampler is not None:
        sampler.set_epoch(0)

    model.eval()

    feats_local: List[np.ndarray] = []
    y_local: List[np.ndarray] = []
    paths_local: List[str] = []
    idxs_local: List[np.ndarray] = []

    for x, y, p, idx in loader:
        x = x.to(device, non_blocking=True)

        if use_proj:
            feat_2048 = model.forward_backbone(x)     # (B,2048) eval_mode=False
            out = model.forward_head(feat_2048)       # (emb, logits) or emb

            if isinstance(out, (tuple, list)):
                feat_out = out[0]                     # emb
            else:
                feat_out = out

            # normalize
            feat_out = F.normalize(feat_out, dim=1)
        else:
            feat_map = model.forward_backbone(x)   # (B,2048,H,W) or (B,2048)
            if feat_map.ndim == 4:
                feat_out = F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)  # (B,2048)
            else:
                feat_out = feat_map.view(feat_map.size(0), -1)
        
        y_local.append(y.cpu().numpy())

        feats_local.append(feat_out.cpu().numpy())
        paths_local.extend(list(p))
        idxs_local.append(idx.cpu().numpy())

    out_dim = int(feat_dim) if use_proj else 2048

    feats_local_np = (
        np.concatenate(feats_local, axis=0)
        if len(feats_local) else np.zeros((0, out_dim), dtype=np.float32)
    )
    y_local_np = np.concatenate(y_local, axis=0) if len(y_local) else None

    idxs_local_np = (
        np.concatenate(idxs_local, axis=0).astype(np.int64)
        if len(idxs_local) else np.zeros((0,), dtype=np.int64)
    )

    # restore local order (important even without DDP)
    if len(idxs_local_np) > 0:
        order = np.argsort(idxs_local_np)
        feats_local_np = feats_local_np[order]
        if y_local_np is not None and len(y_local_np) == len(order):
            y_local_np = y_local_np[order]
        paths_local = [paths_local[i] for i in order.tolist()]
        idxs_local_np = idxs_local_np[order]

    # gather across ranks
    if distributed and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()

        paths_local_np = np.array(paths_local, dtype=object)

        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(
            gathered,
            (feats_local_np, y_local_np, paths_local_np, idxs_local_np)
        )

        feats_all, y_all, idxs_all, paths_all = [], [], [], []

        for f_np, y_np, p_np, idx_np in gathered:
            if f_np is None: 
                continue
            if len(f_np) != len(idx_np) or len(p_np) != len(idx_np):
                raise RuntimeError("gathered lengths mismatch")

            feats_all.append(f_np)
            idxs_all.append(idx_np.astype(np.int64))
            paths_all.append(p_np)

            if y_np is not None and len(y_np) == len(idx_np):
                y_all.append(y_np)

        feats_np = np.concatenate(feats_all, axis=0)
        idxs_np = np.concatenate(idxs_all, axis=0)

        y_true = np.concatenate(y_all, axis=0) if len(y_all) else None
        paths_np = np.concatenate(paths_all, axis=0)

        # restore global order by idxs_np
        order = np.argsort(idxs_np)
        feats_np = feats_np[order]
        idxs_np = idxs_np[order]
        paths_np = paths_np[order]
        if y_true is not None and len(y_true) == len(order):
            y_true = y_true[order]

        return EmbedResult(feats=feats_np, y_true=y_true, paths=paths_np.tolist())

    return EmbedResult(feats=feats_local_np, y_true=y_local_np, paths=paths_local)

# Pair-head: feature fusion (supp Eq.(2)) + linear head
def fuse_features_torch(zp: torch.Tensor, zq: torch.Tensor, fusion: str) -> torch.Tensor:
    if fusion == "concat":
        return torch.cat([zp, zq], dim=1)
    if fusion == "prod":
        return zp * zq
    if fusion == "absdiff":
        return torch.abs(zp - zq)
    if fusion == "l2diff":
        return torch.norm(zp - zq, dim=1, keepdim=True)
    raise ValueError(f"Unknown fusion: {fusion}")

def fused_dim(d: int, fusion: str) -> int:
    if fusion == "concat":
        return 2 * d
    if fusion in ["prod", "absdiff"]:
        return d
    if fusion == "l2diff":
        return 1
    raise ValueError(f"Unknown fusion: {fusion}")

class PairwiseLinearHead(nn.Module):
    def __init__(self, d: int, fusion: str, use_bn: bool = False):
        super().__init__()
        in_dim = fused_dim(int(d), fusion)
        layers = []
        if use_bn:
            layers.append(nn.BatchNorm1d(in_dim))
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.fusion = fusion

    def forward(self, zp: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        x = fuse_features_torch(zp, zq, self.fusion)
        return self.net(x).squeeze(1)  # (B,)

def build_positive_graph_knn(
    feats: np.ndarray,
    k_nn: int,
    m: float,
    metric: str = "cosine",
    mutual: bool = True,
) -> List[set]:
    """
    Build sparse positive-edge adjacency using kNN and similarity threshold m.
    If mutual=True, keep edge (i,j) only if i is in knn(j) AND j is in knn(i).
    For metric="cosine": sklearn returns cosine distance = 1 - cosine_similarity.
    """
    N = feats.shape[0]
    pos_neighbors = [set() for _ in range(N)]
    if N == 0:
        return pos_neighbors

    nn_model = NearestNeighbors(
        n_neighbors=min(k_nn + 1, N), metric=metric, algorithm="auto"
    )
    nn_model.fit(feats)
    dists, inds = nn_model.kneighbors(feats, return_distance=True)

    # build knn sets (exclude self)
    knn_sets = []
    for i in range(N):
        s = set(int(j) for j in inds[i] if int(j) != i)
        knn_sets.append(s)

    for i in range(N):
        for d, j in zip(dists[i], inds[i]):
            j = int(j)
            if j == i:
                continue

            if mutual and (i not in knn_sets[j]):
                continue

            if metric == "cosine":
                sim = 1.0 - float(d)
                if sim >= m:
                    pos_neighbors[i].add(j)
                    pos_neighbors[j].add(i)

    return pos_neighbors

def pivot_correlation_clustering(pos_neighbors: List[set], seed: int = 0) -> np.ndarray:
    """
    Pivot (KwikCluster) algorithm on implicit signed graph:
    - positive edges = pos_neighbors
    - negative edges = all other pairs
    """
    rng = random.Random(seed)
    N = len(pos_neighbors)
    unassigned = set(range(N))
    cluster_id = -np.ones(N, dtype=np.int32)
    cid = 0

    while unassigned:
        pivot = rng.choice(tuple(unassigned))
        cluster = {pivot}
        for nb in pos_neighbors[pivot]:
            if nb in unassigned:
                cluster.add(nb)

        for v in cluster:
            cluster_id[v] = cid
        unassigned -= cluster
        cid += 1

    return cluster_id

# -----------------------------
# ILP solver for correlation clustering (optimal) - small N only
# Variables: x_pq = 1 if CUT (different clusters), 0 if JOIN
# Objective (disagreement):
#   w_pq = logit (pair_head)  OR  w_pq = cos - m  (cosine-margin)
#   w+ = max(w,0), w- = max(-w,0)
#   min sum_{p<q} [ w+ * x_pq + w- * (1 - x_pq) ]
# Constraints (triangle inequalities):
#   x_pq <= x_pr + x_rq  for all distinct p,q,r
# -----------------------------
def solve_cc_ilp_optimal(weights: np.ndarray, backend: str = "pulp") -> np.ndarray:
    """
    weights: (N,N) symmetric, zero diagonal. w_pq > 0 means prefer JOIN, w_pq < 0 means prefer CUT.
    returns: cluster_id array length N (decoded from optimal x_pq via union-find on JOIN edges).
    """
    N = weights.shape[0]
    assert weights.shape == (N, N)

    # lazy import so pivot path doesn't need pulp
    if backend == "pulp":
        import pulp
        prob = pulp.LpProblem("corr_clustering", pulp.LpMinimize)

        # binary vars for p<q
        x = {}
        for p in range(N):
            for q in range(p + 1, N):
                x[(p, q)] = pulp.LpVariable(f"x_{p}_{q}", lowBound=0, upBound=1, cat="Binary")

        def xvar(a, b):
            if a < b:
                return x[(a, b)]
            return x[(b, a)]

        # objective
        obj_terms = []
        for p in range(N):
            for q in range(p + 1, N):
                w = float(weights[p, q])
                wp = max(w, 0.0)      # penalty if cut when should join
                wn = max(-w, 0.0)     # penalty if join when should cut
                # wp * x + wn * (1-x) = (wp-wn)*x + wn
                obj_terms.append((wp - wn) * x[(p, q)] + wn)
        prob += pulp.lpSum(obj_terms)

        # triangle inequalities
        for p in range(N):
            for q in range(p + 1, N):
                for r in range(q + 1, N):
                    prob += xvar(p, q) <= xvar(p, r) + xvar(r, q)
                    prob += xvar(p, r) <= xvar(p, q) + xvar(q, r)
                    prob += xvar(q, r) <= xvar(q, p) + xvar(p, r)

        # solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # decode: join edges are x_pq==0
        parent = list(range(N))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for p in range(N):
            for q in range(p + 1, N):
                val = pulp.value(x[(p, q)])
                if val is None:
                    continue
                if float(val) < 0.5:  # JOIN
                    union(p, q)

        # compress to 0..K-1 ids
        roots = [find(i) for i in range(N)]
        uniq = {}
        cid = 0
        out = np.zeros(N, dtype=np.int32)
        for i, r in enumerate(roots):
            if r not in uniq:
                uniq[r] = cid
                cid += 1
            out[i] = uniq[r]
        return out

    # TODO: gurobi backend if you really need
    raise NotImplementedError("Only pulp backend implemented in this patch.")

# ---------------------------
# Visualization: montage per cluster
# ---------------------------
def save_cluster_montages(
    paths: List[str],
    cluster_ids: np.ndarray,
    out_dir: str,
    max_images_per_cluster: int = 25,
    ncol: int = 5,
    img_size: int = 128,
):
    os.makedirs(out_dir, exist_ok=True)

    clusters: Dict[int, List[int]] = {}
    for i, c in enumerate(cluster_ids.tolist()):
        clusters.setdefault(int(c), []).append(i)

    for c, idxs in clusters.items():
        idxs = idxs[:max_images_per_cluster]
        imgs = []
        for i in idxs:
            img = Image.open(paths[i])
            img = img.convert("RGB")
            img = img.resize((img_size, img_size))
            imgs.append(transforms.ToTensor()(img))
        if not imgs:
            continue
        grid = vutils.make_grid(imgs, nrow=ncol)
        out_path = os.path.join(out_dir, f"cluster_{c:03d}.png")
        vutils.save_image(grid, out_path)


# ---------------------------
# Main
# ---------------------------
parser = argparse.ArgumentParser(description="SwAV-style Correlation Clustering Evaluation")

# data
parser.add_argument("--data_path", type=str, required=True, help="dataset root")
parser.add_argument("--split", type=str, required=True, help="folder inside data_path (e.g. test100/test30/unlabeled)")
parser.add_argument("--extra_split", type=str, default="", help="optional second split to concat with split")
parser.add_argument("--unlabeled", type=bool_flag, default=False, help="treat split as unlabeled flat folder (no class subfolders)")
parser.add_argument("--extra_unlabeled", type=bool_flag, default=False, help="treat extra_split as unlabeled flat folder")

# model / ckpt (keep for ckpt compat)
parser.add_argument("--arch", type=str, default="resnet50", help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int, help="projection head hidden dim (kept for ckpt-compat)")
parser.add_argument("--feat_dim", default=128, type=int, help="SwAV embedding dim (kept for ckpt-compat)")
parser.add_argument("--nmb_prototypes", default=100, type=int, help="kept for ckpt-compat")
parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
parser.add_argument("--use_proj", type=bool_flag, default=True,
                    help="If true, use SwAV projection head embeddings (128-d). Otherwise use backbone 2048-d.")

# eval
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--workers", type=int, default=10)

# dist / misc
parser.add_argument("--dist_url", default="env://", type=str)
parser.add_argument("--world_size", default=-1, type=int)
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--seed", type=int, default=31)

# clustering params
parser.add_argument("--knn", type=int, default=30)

# Two pipelines
parser.add_argument("--pipeline", type=str, default="cosine",
                    choices=["cosine", "pair_head"],
                    help="cosine: use margin m with cosine similarity (supp Eq.3). "
                         "pair_head: use trained pairwise head logits, cost c'=-logit.")

# Cosine-margin (supp Eq.3) params
parser.add_argument("--m", type=float, default=0.55,  # margin
                    help="margin m in supp Eq.(3): c_pq = m - cos(zp,zq). Join iff c_pq < 0 <=> cos > m")
parser.add_argument("--m_min", type=float, default=0.10)
parser.add_argument("--m_max", type=float, default=0.90)
parser.add_argument("--m_steps", type=int, default=81)
parser.add_argument("--select_m_on_train", type=bool_flag, default=False,
                    help="Select margin m on Train-100 by maximizing metric.")
parser.add_argument("--m_select_metric", type=str, default="ACC",
                    choices=["ACC", "F1J", "F1C"],
                    help="Metric used to pick m on train split.")
parser.add_argument("--save_m_sweep", type=bool_flag, default=False,
                    help="Save margin sweep metrics to CSV.")

# Pair-head pipeline params (supp c'=-h)
parser.add_argument("--pair_head_path", type=str, default="",
                    help="Path to trained pair_head.pth.tar (state_dict). Required if pipeline=pair_head.")
parser.add_argument("--fusion", type=str, default="absdiff",
                    choices=["concat", "prod", "absdiff", "l2diff"],
                    help="Feature fusion used by the pair head (must match training).")
parser.add_argument("--logit_thresh", type=float, default=0.0,
                    help="Join iff logit > logit_thresh. Default 0 aligns with supplementary decision rule.")

# Solver choice (mentor: ILP optimal for small N; approx for N=1000)
parser.add_argument("--solver", type=str, default="pivot",
                    choices=["pivot", "ilp"],
                    help="pivot: approximate KwikCluster (good for N~1000). "
                         "ilp: solve to optimality (recommended for N<=130).")
parser.add_argument("--ilp_backend", type=str, default="pulp",
                    choices=["pulp", "gurobi"],
                    help="ILP backend. Use pulp by default. Gurobi if available.")

# outputs
parser.add_argument("--dump_path", type=str, default=".", help="experiment dump path (logs + outputs)")
parser.add_argument("--save_montage", type=bool_flag, default=False)
parser.add_argument("--montage_max", type=int, default=25)
parser.add_argument("--montage_cols", type=int, default=5)

# visualization
parser.add_argument("--save_two_row", type=bool_flag, default=False,
                    help="Save a 2-row figure: Row1 originals, Row2 clustered with yellow boxes.")
parser.add_argument("--two_row_cols", type=int, default=20,
                    help="Number of columns in the 2-row figure grid.")
parser.add_argument("--two_row_img_size", type=int, default=64,
                    help="Tile size (pixels) for each small image in the 2-row figure.")
parser.add_argument("--two_row_pad", type=int, default=2,
                    help="Padding (pixels) between tiles in the 2-row figure.")
parser.add_argument("--two_row_gap", type=int, default=12,
                    help="Vertical gap (pixels) between Row1 and Row2.")


def main():
    args = parser.parse_args()

    # IMPORTANT: make dump_path before initialize_exp (it writes params.pkl)
    os.makedirs(args.dump_path, exist_ok=True)

    # distributed init (your utils handles both torchrun and single-process)
    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    # init exp (logger + stats)
    logger_, _stats = initialize_exp(args, "N", "num_clusters", "ACC", "PJ", "RJ", "PC", "RC", "RI", "VI", "purity", "nmi", "ari")

    # device (utils guarantees args.gpu_to_work_on exists)
    if torch.cuda.is_available() and args.gpu_to_work_on >= 0:
        device = f"cuda:{args.gpu_to_work_on}"
    else:
        device = "cpu"
    logger_.info(f"[device] {device}")

    # dataset
    split_path = os.path.join(args.data_path, args.split)
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Split folder not found: {split_path}")

    ds_main = build_dataset(split_path, img_size=args.img_size, unlabeled=args.unlabeled)

    dataset = ds_main
    split_name = args.split
    if args.extra_split:
        extra_path = os.path.join(args.data_path, args.extra_split)
        if not os.path.isdir(extra_path):
            raise FileNotFoundError(f"Extra split folder not found: {extra_path}")
        ds_extra = build_dataset(extra_path, img_size=args.img_size, unlabeled=args.extra_unlabeled)
        dataset = ConcatDataset([ds_main, ds_extra])
        split_name = f"{args.split}+{args.extra_split}"

    dataset = IndexedDataset(dataset)

    logger_.info(f"[data] total images = {len(dataset)}")

    # ==========================================================================
    if args.use_proj:
        model = resnet_models.__dict__[args.arch](
            normalize=True,
            hidden_mlp=args.hidden_mlp,
            output_dim=args.feat_dim,          # 128
            nmb_prototypes=args.nmb_prototypes
        ).to(device)
    else:
        model = resnet_models.__dict__[args.arch](output_dim=0, eval_mode=True).to(device)

    # ==========================================================================
    model.eval()

    # load weights (same as linear eval)
    if os.path.isfile(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}

        if not args.use_proj:
            # 2048
            state = {k: v for k, v in state.items()
                    if not (k.startswith("projection_head.") or k.startswith("prototypes."))}

        msg = model.load_state_dict(state, strict=False)
        logger_.info(f"Load msg: {msg}")

        # check
        missing = [k for k in msg.missing_keys
                if not (k.startswith("projection_head.") or k.startswith("prototypes."))]
        if len(missing) > 0:
            raise RuntimeError(f"Backbone weights missing in ckpt: {missing[:10]} ...")

    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model.eval()

    # distributed flag (match your utils behavior)
    distributed = (args.world_size > 1 and dist.is_available() and dist.is_initialized())

    # extract embeddings
    emb = extract_embeddings(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        distributed=distributed,
        use_proj=args.use_proj,
        feat_dim=args.feat_dim,
    )

    # only rank0 writes / clusters
    if distributed and args.rank != 0:
        return
    
    summarize_features(f"feat_dim{emb.feats.shape[1]}", emb.feats, logger_)
    feats = l2_normalize(emb.feats.astype(np.float32))

    N = int(feats.shape[0])
    logger_.info(f"[embed] feats shape = {tuple(feats.shape)}")

    
    # save outputs
    out_dir = args.dump_path
    os.makedirs(out_dir, exist_ok=True)
    safe_name = split_name.replace("/", "_")

    # only possible if labels exist
    has_labels = (not args.unlabeled)
    y_true = emb.y_true.astype(np.int32) if has_labels else None

    # build full cosine similarity matrix only when needed
    sim = cosine_similarity_matrix(feats) if has_labels else None

    # ==========================================================
    if has_labels:
        iu = np.triu_indices(N, k=1)
        s = sim[iu]
        logger_.info(
            f"[sim] cosine: min={s.min():.4f} p1={np.percentile(s,1):.4f} "
            f"p50={np.percentile(s,50):.4f} p99={np.percentile(s,99):.4f} max={s.max():.4f}"
        )
        # ---- Save cosine similarity histogram (supp Cosine Similarity (1)) ----
        hist_path = os.path.join(out_dir, f"cosine_hist_{safe_name}.png")
        save_cosine_hist(
            sim=sim,
            out_path=hist_path,
            title=f"Pairwise cosine similarities: {split_name} (N={N})",
            bins=80,
        )
        logger_.info(f"[save] cosine hist: {hist_path}")
    # ==========================================================

    def eval_cosine_at_m(m_val: float) -> Dict[str, float]:
        pos_graph = build_positive_graph_knn(feats=feats, k_nn=args.knn, m=m_val, metric="cosine", mutual=True)
        y_pred = pivot_correlation_clustering(pos_graph, seed=args.seed)
        out = {"m": float(m_val), "num_clusters": int(len(np.unique(y_pred)))}
        if has_labels:
            out.update(pairwise_metrics_from_similarity(sim=sim, y_true=y_true, m=m_val))
            out["RI"] = float(rand_index_score(y_true, y_pred.astype(np.int32)))
            out["VI"] = float(variation_of_information(y_true, y_pred.astype(np.int32)))
        return out

    # =========================
    # PIPELINE SWITCH
    # =========================
    final_row = {}
    cluster_ids = None

    if args.pipeline == "cosine":
        m_used = float(args.m)
        m_rows: List[Dict[str, float]] = []

        if has_labels and (args.save_m_sweep or args.select_m_on_train):
            ms = np.linspace(args.m_min, args.m_max, args.m_steps, dtype=np.float64)

            # ---- Task (2) strict: select m by maximizing ACC using thresholded pairwise decisions ----
            select_metric = "ACC" if args.select_m_on_train else args.m_select_metric

            best_m, best_score = None, -1e9
            for mv in ms:
                mv = float(mv)
                row = {"m": mv}
                row.update(pairwise_metrics_from_similarity(sim=sim, y_true=y_true, m=mv))
                m_rows.append(row)

                score = row.get(select_metric, None)
                if score is not None and float(score) > best_score:
                    best_score = float(score)
                    best_m = mv

            if args.select_m_on_train and best_m is not None:
                m_used = float(best_m)
                logger_.info(f"[m] selected m={m_used:.4f} by ACC (best={best_score:.4f})")

        # save m sweep (csv + plot) regardless of solver
        if has_labels and args.save_m_sweep and len(m_rows) > 0:
            m_csv = os.path.join(args.dump_path, f"m_sweep_{safe_name}.csv")
            keys = list(m_rows[0].keys())
            with open(m_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in m_rows:
                    w.writerow(r)
            logger_.info(f"[save] m sweep csv: {m_csv}")

            metric_name = ("ACC" if args.select_m_on_train else args.m_select_metric)
            m_plot = os.path.join(args.dump_path, f"m_sweep_{safe_name}_{metric_name}.png")
            save_m_sweep_plot(
                m_rows=m_rows,
                metric=metric_name,
                out_path=m_plot,
                title=f"Margin sweep on {split_name}",
            )
            logger_.info(f"[save] m sweep plot: {m_plot}")

        # --- Solve clustering with chosen solver ---
        if args.solver == "ilp":
            # Need full sim matrix (O(N^2)); only realistic for small N
            if sim is None:
                # unlabeled still can compute sim from feats
                sim_full = cosine_similarity_matrix(feats)
            else:
                sim_full = sim

            weights = cosine_margin_weights(sim_full, m=m_used)  # w = cos - m
            cluster_ids = solve_cc_ilp_optimal(weights=weights, backend=args.ilp_backend)

            final_row = {"m": float(m_used), "num_clusters": int(len(np.unique(cluster_ids)))}

            if has_labels:
                # pairwise metrics use the same decision rule: join iff cos >= m
                final_row.update(pairwise_metrics_from_similarity(sim=sim_full, y_true=y_true, m=m_used))
                final_row["RI"] = float(rand_index_score(y_true, cluster_ids.astype(np.int32)))
                final_row["VI"] = float(variation_of_information(y_true, cluster_ids.astype(np.int32)))

            logger_.info(
                f"[cluster/cosine+ilp] m={m_used:.4f} num_clusters={final_row['num_clusters']} "
                + (f"ACC={final_row.get('ACC', None)} RI={final_row.get('RI', None)} VI={final_row.get('VI', None)}" if has_labels else "")
            )

        else:
            # pivot approximate
            final_row = eval_cosine_at_m(m_used)
            pos_graph = build_positive_graph_knn(feats=feats, k_nn=args.knn, m=m_used, metric="cosine", mutual=True)
            cluster_ids = pivot_correlation_clustering(pos_graph, seed=args.seed)

            logger_.info(
                f"[cluster/cosine+pivot] m={m_used:.4f} num_clusters={final_row['num_clusters']} "
                + (f"ACC={final_row.get('ACC', None)} RI={final_row.get('RI', None)} VI={final_row.get('VI', None)}" if has_labels else "")
            )

    else:
        # =========================
        # PAIR_HEAD PIPELINE
        # cost: c'_{pq} = -logit
        # join rule: logit > 0 (default)
        # haven't be solved. There is not enough time.
        # =========================
        if args.pair_head_path == "":
            raise ValueError("pipeline=pair_head requires --pair_head_path")

        head = PairwiseLinearHead(d=args.feat_dim, fusion=args.fusion, use_bn=False).to(device)
        ck = torch.load(args.pair_head_path, map_location=device)
        state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        state = {k.replace("module.", ""): v for k, v in state.items()}
        head.load_state_dict(state, strict=True)
        head.eval()

        feats_t = torch.from_numpy(feats).to(device)  # (N,d)

        if args.solver == "ilp":
            # full NxN logits (O(N^2)) -> only for small N
            logits_mat = np.zeros((N, N), dtype=np.float32)

            with torch.no_grad():
                for p in range(N):
                    zq = feats_t[p + 1:]
                    if zq.numel() == 0:
                        continue
                    zp = feats_t[p].unsqueeze(0).expand(zq.size(0), -1)
                    logits = head(zp, zq).detach().cpu().numpy().astype(np.float32)
                    logits_mat[p, p + 1:] = logits
                    logits_mat[p + 1:, p] = logits

            # ILP weights: w_pq = logit (w>0 join, w<0 cut)
            cluster_ids = solve_cc_ilp_optimal(weights=logits_mat, backend=args.ilp_backend)
            final_row = {"num_clusters": int(len(np.unique(cluster_ids)))}

            if has_labels:
                final_row.update(pairwise_metrics_from_logits(logits=logits_mat, y_true=y_true, logit_thresh=args.logit_thresh))
                final_row["RI"] = float(rand_index_score(y_true, cluster_ids.astype(np.int32)))
                final_row["VI"] = float(variation_of_information(y_true, cluster_ids.astype(np.int32)))

        else:
            # pivot approx: mutual knn candidates + logit sign
            nn_model = NearestNeighbors(n_neighbors=min(args.knn + 1, N), metric="cosine")
            nn_model.fit(feats)
            _, inds = nn_model.kneighbors(feats, return_distance=True)

            # build mutual knn sets (exclude self)
            knn_sets = [set(int(j) for j in inds[i] if int(j) != i) for i in range(N)]

            pos_neighbors = [set() for _ in range(N)]

            # (optional) store sparse logits for metrics on labeled splits
            logits_mat = np.zeros((N, N), dtype=np.float32) if has_labels else None

            with torch.no_grad():
                for i in range(N):
                    cand = [int(j) for j in inds[i] if int(j) != i]
                    if not cand:
                        continue

                    cand = [j for j in cand if i in knn_sets[j]]  # mutual filter
                    if not cand:
                        continue

                    zp = feats_t[i].unsqueeze(0).expand(len(cand), -1)
                    zq = feats_t[cand]
                    logits = head(zp, zq).detach().cpu().numpy().astype(np.float32)

                    for j, lj in zip(cand, logits.tolist()):
                        lj = float(lj)
                        if logits_mat is not None:
                            logits_mat[i, j] = lj
                            logits_mat[j, i] = lj
                        if lj > float(args.logit_thresh):
                            pos_neighbors[i].add(int(j))
                            pos_neighbors[int(j)].add(int(i))

            cluster_ids = pivot_correlation_clustering(pos_neighbors, seed=args.seed)
            final_row = {"num_clusters": int(len(np.unique(cluster_ids)))}

            if has_labels:
                # IMPORTANT:
                # Here logits_mat only contains kNN edges; non-edges are left as 0.
                # That means pairwise metrics are an approximation of the pair-head classifier.
                final_row.update(pairwise_metrics_from_logits(logits=logits_mat, y_true=y_true, logit_thresh=args.logit_thresh))
                final_row["RI"] = float(rand_index_score(y_true, cluster_ids.astype(np.int32)))
                final_row["VI"] = float(variation_of_information(y_true, cluster_ids.astype(np.int32)))

        logger_.info(f"[cluster/pair_head] solver={args.solver} thresh={args.logit_thresh} "
                    f"num_clusters={final_row['num_clusters']} "
                    + (f"ACC={final_row.get('ACC', None)} RI={final_row.get('RI', None)} VI={final_row.get('VI', None)}" if has_labels else ""))

    assert cluster_ids is not None

    # clusters csv
    csv_path = os.path.join(out_dir, f"clusters_{safe_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "cluster_id", "true_label"])
        if emb.y_true is None:
            for p, c in zip(emb.paths, cluster_ids.tolist()):
                writer.writerow([p, int(c), ""])
        else:
            for p, c, y in zip(emb.paths, cluster_ids.tolist(), emb.y_true.tolist()):
                writer.writerow([p, int(c), int(y)])

    # metrics json (paper-aligned first)
    metrics = {
        "pipeline": args.pipeline,
        "solver": args.solver,  # cosine only
        "N": N,
        "knn": int(args.knn),
        "num_clusters": int(final_row["num_clusters"]),
        "ACC": final_row.get("ACC", None),
        "PC": final_row.get("PC", None),
        "RC": final_row.get("RC", None),
        "PJ": final_row.get("PJ", None),
        "RJ": final_row.get("RJ", None),
        "F1C": final_row.get("F1C", None),
        "F1J": final_row.get("F1J", None),
        "RI": final_row.get("RI", None),
        "VI": final_row.get("VI", None),
    }

    # pipeline-specific fields
    if args.pipeline == "cosine":
        metrics["m"] = float(m_used)
    else:
        metrics["pair_head_path"] = args.pair_head_path
        metrics["fusion"] = args.fusion
        metrics["logit_thresh"] = float(args.logit_thresh)
        metrics["ilp_backend"] = (args.ilp_backend if args.solver == "ilp" else None)

    if has_labels:
        assert y_true is not None
        y_pred = cluster_ids.astype(np.int32)
        metrics["purity"] = float(purity_score(y_true, y_pred))
        metrics["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
        metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
    else:
        metrics["purity"] = None
        metrics["nmi"] = None
        metrics["ari"] = None

    json_path = os.path.join(out_dir, f"metrics_{safe_name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger_.info(f"[save] clusters csv: {csv_path}")
    logger_.info(f"[save] metrics json: {json_path}")
    logger_.info(f"[metrics] {metrics}")

    # montage
    if args.save_montage:
        montage_dir = os.path.join(out_dir, f"montages_{safe_name}")
        save_cluster_montages(
            paths=emb.paths,
            cluster_ids=cluster_ids,
            out_dir=montage_dir,
            max_images_per_cluster=args.montage_max,
            ncol=args.montage_cols,
            img_size=128,
        )
        logger_.info(f"[save] montages: {montage_dir}")

    # cluster visualization
    if args.save_two_row:
        fig_path = os.path.join(out_dir, f"two_row_{safe_name}.png")
        save_two_row_figure(
            paths=emb.paths,
            cluster_ids=cluster_ids,
            true_labels=(emb.y_true.astype(np.int32) if has_labels else None),
            out_path=fig_path,
            ncols=int(args.two_row_cols),
            tile=int(args.two_row_img_size),
            pad=int(args.two_row_pad),
            row_gap=int(args.two_row_gap),
        )
        logger_.info(f"[save] two-row figure: {fig_path}")


if __name__ == "__main__":
    main()
