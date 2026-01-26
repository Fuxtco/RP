# Train pairwise linear head h_nu on train_100, aligned with supplementary material.
# - Frozen embedding network g_{lambda_hat}: SwAV 128-d embedding (forward_backbone -> forward_head)
# - Pair feature fusion: absdiff (Eq. (2)) by default
# - Decision rule later: join iff logit > 0

import argparse
import os
import time
import random
from logging import getLogger
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image

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
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)

import src.resnet50 as resnet_models

logger = getLogger()


# -----------------------------
# Dist helpers
# -----------------------------
def _is_dist():
    return dist.is_available() and dist.is_initialized()

def _rank():
    return dist.get_rank() if _is_dist() else 0

def is_rank0():
    return _rank() == 0


# -----------------------------
# Data: labeled ImageFolder
# -----------------------------
class LabeledImageFolder(datasets.ImageFolder):
    """ImageFolder but returns (img, label)."""
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = self.loader(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(target)


def build_train_transform(img_size: int = 224):
    mean = [0.0832, 0.0897, 0.0733]
    std  = [0.0976, 0.1375, 0.0852]
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# -----------------------------
# Pair datasets
# -----------------------------
class AllPairsDataset(data.Dataset):
    """Enumerate all unordered pairs (p,q), p<q. Suitable for N~100."""
    def __init__(self, base_ds: data.Dataset):
        self.base = base_ds
        self.N = len(base_ds)
        if self.N < 2:
            raise RuntimeError("Need at least 2 samples to build pairs.")
        self.M = self.N * (self.N - 1) // 2

        row_counts = np.array([self.N - 1 - p for p in range(self.N - 1)], dtype=np.int64)
        self.row_offsets = np.concatenate([[0], np.cumsum(row_counts)], axis=0)  # len N

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


# -----------------------------
# Embedding g_{lambda_hat}
# -----------------------------
@torch.no_grad()
def compute_embedding_g(model: nn.Module, x: torch.Tensor, use_proj: bool) -> torch.Tensor:
    """
    use_proj=True:
      z = forward_head(forward_backbone(x))[0] or out
      NOTE: if model built with normalize=True, z is already L2-normalized.
    """
    if use_proj:
        feat_2048 = model.forward_backbone(x)
        out = model.forward_head(feat_2048)
        z = out[0] if isinstance(out, (tuple, list)) else out
        return z

    # 2048 fallback (not recommended for your report)
    feat = model.forward_backbone(x)
    if feat.ndim == 4:
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
    else:
        feat = feat.view(feat.size(0), -1)
    return feat


# -----------------------------
# Feature fusion (supp Eq. (1) and (2))
# -----------------------------
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
    """Linear head h_nu: logit -> join iff logit > 0."""
    def __init__(self, d: int, fusion: str, use_bn: bool = False):
        super().__init__()
        in_dim = fused_dim(int(d), fusion)
        self.fusion = fusion

        layers = []
        if use_bn:
            layers.append(nn.BatchNorm1d(in_dim))
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

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


def train_one_epoch(model_g: nn.Module, head: nn.Module, optimizer, loader, epoch: int,
                    use_proj: bool, log_interval: int = 100):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_m = AverageMeter()
    f1j_m = AverageMeter()
    f1c_m = AverageMeter()

    model_g.eval()
    head.train()
    criterion = nn.BCEWithLogitsLoss().cuda()

    end = time.perf_counter()
    for it, (x1, x2, y_same) in enumerate(loader):
        data_time.update(time.perf_counter() - end)

        x1 = x1.cuda(non_blocking=True)
        x2 = x2.cuda(non_blocking=True)
        y_same = y_same.cuda(non_blocking=True)

        with torch.no_grad():
            z1 = compute_embedding_g(model_g, x1, use_proj=use_proj)
            z2 = compute_embedding_g(model_g, x2, use_proj=use_proj)

        logits = head(z1, z2)
        loss = criterion(logits, y_same)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), x1.size(0))
        m = pairwise_metrics_from_logits(logits.detach(), y_same.detach())
        acc_m.update(m["ACC"], x1.size(0))
        f1j_m.update(m["F1J"], x1.size(0))
        f1c_m.update(m["F1C"], x1.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if is_rank0() and it % log_interval == 0:
            logger.info(
                "Epoch[{0}] Iter[{1}/{2}] "
                "Time {bt.val:.3f} ({bt.avg:.3f}) "
                "Data {dt.val:.3f} ({dt.avg:.3f}) "
                "Loss {ls.val:.4f} ({ls.avg:.4f}) "
                "ACC {acc.val:.4f} ({acc.avg:.4f}) "
                "F1J {f1j.val:.4f} ({f1j.avg:.4f}) "
                "F1C {f1c.val:.4f} ({f1c.avg:.4f}) "
                "LR {lr:.6f}".format(
                    epoch, it, len(loader),
                    bt=batch_time, dt=data_time, ls=losses,
                    acc=acc_m, f1j=f1j_m, f1c=f1c_m,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

    return {"loss": losses.avg, "ACC": acc_m.avg, "F1J": f1j_m.avg, "F1C": f1c_m.avg}


# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser("Train pairwise linear head (supplementary)")

parser.add_argument("--dump_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=31)

parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--train_split", type=str, default="train_100")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--workers", type=int, default=10)

parser.add_argument("--arch", type=str, default="resnet50")
parser.add_argument("--hidden_mlp", default=2048, type=int)
parser.add_argument("--feat_dim", default=128, type=int)
parser.add_argument("--nmb_prototypes", default=100, type=int)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--use_proj", type=bool_flag, default=True)

parser.add_argument("--fusion", type=str, default="absdiff",
                    choices=["concat", "prod", "absdiff", "l2diff"])
parser.add_argument("--use_bn", type=bool_flag, default=False)

parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=256, help="pairs per batch")
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--wd", type=float, default=1e-6)
parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["cosine", "step"])
parser.add_argument("--decay_epochs", type=int, nargs="+", default=[30, 40])
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--final_lr", type=float, default=0.0)

parser.add_argument("--dist_url", default="env://", type=str)
parser.add_argument("--world_size", default=-1, type=int)
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--local_rank", default=0, type=int)


def main():
    args = parser.parse_args()
    os.makedirs(args.dump_path, exist_ok=True)

    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    logger_, stats = initialize_exp(
        args, "epoch", "loss", "ACC", "F1J", "F1C"
    )
    cudnn.benchmark = True

    # dataset
    train_path = os.path.join(args.data_path, args.train_split)
    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"train split not found: {train_path}")

    base_train = LabeledImageFolder(train_path, transform=build_train_transform(args.img_size))
    train_ds = AllPairsDataset(base_train)

    if _is_dist():
        train_sampler = data.distributed.DistributedSampler(train_ds, shuffle=True, drop_last=False)
    else:
        train_sampler = None

    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    if is_rank0():
        logger.info(f"[data] train base N={len(base_train)}, pairs={len(train_ds)}")

    # build frozen embedding model g_{lambda_hat}
    if args.use_proj:
        model_g = resnet_models.__dict__[args.arch](
            normalize=True,
            hidden_mlp=args.hidden_mlp,
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
        ).cuda()
    else:
        model_g = resnet_models.__dict__[args.arch](output_dim=0, eval_mode=True).cuda()

    model_g.eval()
    for p in model_g.parameters():
        p.requires_grad_(False)

    # load ckpt
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cuda")
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    if not args.use_proj:
        state = {k: v for k, v in state.items()
                 if not (k.startswith("projection_head.") or k.startswith("prototypes."))}
    msg = model_g.load_state_dict(state, strict=False)
    if is_rank0():
        logger.info(f"[ckpt] load msg: {msg}")

    # head
    d = args.feat_dim if args.use_proj else 2048
    head = PairwiseLinearHead(d=d, fusion=args.fusion, use_bn=args.use_bn).cuda()

    if _is_dist():
        head = nn.SyncBatchNorm.convert_sync_batchnorm(head)
        head = nn.parallel.DistributedDataParallel(
            head,
            device_ids=[args.gpu_to_work_on],
            find_unused_parameters=True,
        )

    optimizer = torch.optim.SGD(head.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=False)

    if args.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.final_lr)

    # resume support
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "pair_head.pth.tar"),
        run_variables=to_restore,
        state_dict=head,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = int(to_restore["epoch"])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_rank0():
            logger.info(f"============ Starting epoch {epoch} ============")

        tr = train_one_epoch(model_g, head, optimizer, train_loader, epoch,
                             use_proj=args.use_proj, log_interval=100)

        stats.update((epoch, tr["loss"], tr["ACC"], tr["F1J"], tr["F1C"]))
        scheduler.step()

        if is_rank0():
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
            }
            torch.save(save_dict, os.path.join(args.dump_path, "pair_head.pth.tar"))

    if is_rank0():
        logger.info("Training completed. Saved: pair_head.pth.tar")


if __name__ == "__main__":
    main()
