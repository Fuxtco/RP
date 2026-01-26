# Evaluate trained pairwise head h_nu on a labeled split (test_100 / test_30 / test_130).
# - Loads frozen g_{lambda_hat} from SwAV ckpt (128-d embedding)
# - Loads h_nu from head checkpoint
# - Builds all unordered pairs C(V,2) and computes pairwise metrics:
#   ACC, PJ/RJ/F1J, PC/RC/F1C

import argparse
import os
import time
import json
from logging import getLogger
from typing import Dict, Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.utils import (
    bool_flag,
    initialize_exp,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)

import src.resnet50 as resnet_models

logger = getLogger()


def _is_dist():
    return dist.is_available() and dist.is_initialized()

def _rank():
    return dist.get_rank() if _is_dist() else 0

def is_rank0():
    return _rank() == 0


class LabeledImageFolder(datasets.ImageFolder):
    """ImageFolder but returns (img, label)."""
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = self.loader(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(target)


def build_eval_transform(img_size: int = 224):
    mean = [0.0832, 0.0897, 0.0733]
    std  = [0.0976, 0.1375, 0.0852]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class AllPairsDataset(data.Dataset):
    """Enumerate all unordered pairs (p,q), p<q."""
    def __init__(self, base_ds: data.Dataset):
        self.base = base_ds
        self.N = len(base_ds)
        if self.N < 2:
            raise RuntimeError("Need at least 2 samples to build pairs.")
        self.M = self.N * (self.N - 1) // 2

        row_counts = np.array([self.N - 1 - p for p in range(self.N - 1)], dtype=np.int64)
        self.row_offsets = np.concatenate([[0], np.cumsum(row_counts)], axis=0)

    def __len__(self):
        return int(self.M)

    def _pair_from_index(self, k: int) -> Tuple[int, int]:
        p = int(np.searchsorted(self.row_offsets, k, side="right") - 1)
        within = k - int(self.row_offsets[p])
        q = p + 1 + int(within)
        return p, q

    def __getitem__(self, idx: int):
        p, q = self._pair_from_index(int(idx))
        x_p, y_p = self.base[p]
        x_q, y_q = self.base[q]
        y_same = 1.0 if (y_p == y_q) else 0.0
        return x_p, x_q, torch.tensor(y_same, dtype=torch.float32)


@torch.no_grad()
def compute_embedding_g(model: nn.Module, x: torch.Tensor, use_proj: bool) -> torch.Tensor:
    if use_proj:
        feat_2048 = model.forward_backbone(x)
        out = model.forward_head(feat_2048)
        z = out[0] if isinstance(out, (tuple, list)) else out
        return z
    feat = model.forward_backbone(x)
    if feat.ndim == 4:
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
    else:
        feat = feat.view(feat.size(0), -1)
    return feat


def fuse_features(zp: torch.Tensor, zq: torch.Tensor, fusion: str) -> torch.Tensor:
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
        self.fusion = fusion

        layers = []
        if use_bn:
            layers.append(nn.BatchNorm1d(in_dim))
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, zp: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        x = fuse_features(zp, zq, self.fusion)
        return self.net(x).squeeze(1)


@torch.no_grad()
def pairwise_metrics_from_logits(logits: torch.Tensor, y_same: torch.Tensor) -> Dict[str, float]:
    y_same = y_same.long()
    pred_join = (logits > 0).long()
    true_join = y_same
    true_cut = 1 - true_join
    pred_cut = 1 - pred_join

    acc = (pred_join == true_join).float().mean().item()

    tpJ = ((pred_join == 1) & (true_join == 1)).sum().item()
    fpJ = ((pred_join == 1) & (true_cut == 1)).sum().item()
    fnJ = ((pred_join == 0) & (true_join == 1)).sum().item()
    PJ = tpJ / (tpJ + fpJ + 1e-12)
    RJ = tpJ / (tpJ + fnJ + 1e-12)
    F1J = 2.0 * PJ * RJ / (PJ + RJ + 1e-12)

    tpC = ((pred_cut == 1) & (true_cut == 1)).sum().item()
    fpC = ((pred_cut == 1) & (true_join == 1)).sum().item()
    fnC = ((pred_cut == 0) & (true_cut == 1)).sum().item()
    PC = tpC / (tpC + fpC + 1e-12)
    RC = tpC / (tpC + fnC + 1e-12)
    F1C = 2.0 * PC * RC / (PC + RC + 1e-12)

    return {"ACC": acc, "PJ": PJ, "RJ": RJ, "F1J": F1J, "PC": PC, "RC": RC, "F1C": F1C}


@torch.no_grad()
def eval_pairs(model_g: nn.Module, head: nn.Module, loader) -> Dict[str, float]:
    losses = AverageMeter()
    acc_m = AverageMeter()
    f1j_m = AverageMeter()
    f1c_m = AverageMeter()
    criterion = nn.BCEWithLogitsLoss().cuda()

    model_g.eval()
    head.eval()

    for x1, x2, y_same in loader:
        x1 = x1.cuda(non_blocking=True)
        x2 = x2.cuda(non_blocking=True)
        y_same = y_same.cuda(non_blocking=True)

        z1 = compute_embedding_g(model_g, x1, use_proj=True)
        z2 = compute_embedding_g(model_g, x2, use_proj=True)

        logits = head(z1, z2)
        loss = criterion(logits, y_same)

        losses.update(loss.item(), x1.size(0))
        m = pairwise_metrics_from_logits(logits, y_same)
        acc_m.update(m["ACC"], x1.size(0))
        f1j_m.update(m["F1J"], x1.size(0))
        f1c_m.update(m["F1C"], x1.size(0))

    return {"loss": losses.avg, "ACC": acc_m.avg, "F1J": f1j_m.avg, "F1C": f1c_m.avg}


parser = argparse.ArgumentParser("Eval pairwise head (supplementary)")

parser.add_argument("--dump_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=31)

parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--split", type=str, required=True, help="test_100 / test_30 / test_130")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--workers", type=int, default=10)

parser.add_argument("--arch", type=str, default="resnet50")
parser.add_argument("--hidden_mlp", default=2048, type=int)
parser.add_argument("--feat_dim", default=128, type=int)
parser.add_argument("--nmb_prototypes", default=100, type=int)
parser.add_argument("--ckpt", type=str, required=True)

parser.add_argument("--head_path", type=str, required=True, help="path to pair_head.pth.tar")
parser.add_argument("--fusion", type=str, default="absdiff",
                    choices=["concat", "prod", "absdiff", "l2diff"])
parser.add_argument("--use_bn", type=bool_flag, default=False)

parser.add_argument("--batch_size", type=int, default=256, help="pairs per batch")

parser.add_argument("--dist_url", default="env://", type=str)
parser.add_argument("--world_size", default=-1, type=int)
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--local_rank", default=0, type=int)


def main():
    args = parser.parse_args()
    os.makedirs(args.dump_path, exist_ok=True)

    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    logger_, _stats = initialize_exp(args, "split", "pairs", "loss", "ACC", "F1J", "F1C")
    cudnn.benchmark = True

    split_path = os.path.join(args.data_path, args.split)
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"split not found: {split_path}")

    base = LabeledImageFolder(split_path, transform=build_eval_transform(args.img_size))
    pair_ds = AllPairsDataset(base)

    loader = data.DataLoader(
        pair_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    if is_rank0():
        logger.info(f"[data] split={args.split} base N={len(base)} pairs={len(pair_ds)}")

    # frozen g
    model_g = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
    ).cuda()
    model_g.eval()
    for p in model_g.parameters():
        p.requires_grad_(False)

    # load swav ckpt
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"SwAV ckpt not found: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cuda")
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    msg = model_g.load_state_dict(state, strict=False)
    if is_rank0():
        logger.info(f"[ckpt] swav load msg: {msg}")

    # load head
    head = PairwiseLinearHead(d=args.feat_dim, fusion=args.fusion, use_bn=args.use_bn).cuda()
    if not os.path.isfile(args.head_path):
        raise FileNotFoundError(f"head ckpt not found: {args.head_path}")
    head_ckpt = torch.load(args.head_path, map_location="cuda")
    head_state = head_ckpt["state_dict"] if "state_dict" in head_ckpt else head_ckpt
    # strip possible "module."
    head_state = {k.replace("module.", ""): v for k, v in head_state.items()}
    head.load_state_dict(head_state, strict=True)

    out = eval_pairs(model_g, head, loader)

    metrics = {
        "split": args.split,
        "N": int(len(base)),
        "pairs": int(len(pair_ds)),
        "fusion": args.fusion,
        **out,
    }

    if is_rank0():
        logger.info(f"[metrics] {metrics}")
        json_path = os.path.join(args.dump_path, f"metrics_{args.split}.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"[save] {json_path}")


if __name__ == "__main__":
    main()
