# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim

'''
import apex
from apex.parallel.LARC import LARC
'''
from src.larc import LARC
from torch.cuda.amp import GradScaler, autocast

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from src.multicropdataset import MultiCropDataset
import src.resnet50 as resnet_models

logger = getLogger()

## 定义一个命令行参数解析器，这个脚本的所有超参数都通过它来管理 ##
parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=100, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")

# apex相关，需要替换
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")


parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")

def is_master():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

@torch.no_grad()
def dump_debug(payload: dict, dump_dir: str, tag: str):
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"debug_{tag}.pth")
    torch.save(payload, path)
    if is_master():
        print(f"[DEBUG] dumped to: {path}", flush=True)

def assert_finite(t: torch.Tensor, name: str, iteration: int, dump_dir: str, extra: dict = None):
    """No spam: only acts when non-finite appears."""
    if t is None:
        return
    if torch.isfinite(t).all():
        return

    # minimal stats
    finite = torch.isfinite(t)
    bad_cnt = (~finite).sum().item()
    tf = t[finite]
    stats = {
        "name": name,
        "iter": iteration,
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "bad_count": bad_cnt,
    }
    if tf.numel() > 0:
        stats.update({
            "finite_min": tf.min().item(),
            "finite_max": tf.max().item(),
            "finite_mean": tf.mean().item(),
        })

    if is_master():
        print(f"[NaN/Inf] {stats}", flush=True)

    # dump only small slices to avoid huge files
    payload = {"stats": stats}
    if extra is not None:
        payload["extra"] = extra
    dump_debug(payload, dump_dir, tag=f"{iteration}_{name}")

    raise RuntimeError(f"Non-finite tensor: {name} at iter {iteration}")


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    args.dump_checkpoints = os.path.join(args.dump_path, "checkpoints")
    os.makedirs(args.dump_checkpoints, exist_ok=True)

    # build data
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    # DataLoader自动调用train_dataset[i]
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
    )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #涉及apex需要替换
    elif args.sync_bn == "apex":
        '''
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
        '''
        raise ValueError("Apex SyncBN is not available. Please use --sync_bn pytorch or none.")
    # copy model to GPU
    model = model.cuda()
    
    # ---- register a few grad hooks to locate the first module producing non-finite grads ----
    bad_grad_first = {"name": None}
    
    def mark_first_bad_grad(name):
        def _hook(grad):
            if grad is not None and (not torch.isfinite(grad).all()) and bad_grad_first["name"] is None:
                bad_grad_first["name"] = name
            return grad
        return _hook
    
    if args.rank == 0:
        # conv1
        model.conv1.weight.register_hook(mark_first_bad_grad("conv1.weight"))
    
        # projection head / fc (depends on your resnet50 implementation)
        if hasattr(model, "fc") and hasattr(model.fc, "weight"):
            model.fc.weight.register_hook(mark_first_bad_grad("fc.weight"))
    
        # prototypes
        if hasattr(model, "prototypes") and hasattr(model.prototypes, "weight"):
            model.prototypes.weight.register_hook(mark_first_bad_grad("prototypes.weight"))
            
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    '''
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")
    '''
    # (torch.cuda.amp)
    scaler = GradScaler(enabled=args.use_fp16, init_scale = 2048.0)
    if args.use_fp16:
        logger.info("Initializing mixed precision with torch.cuda.amp done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        # amp=apex.amp,
        scaler=scaler,
    )
    start_epoch = to_restore["epoch"]

    # build the queue
    queue = None
    queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()

        # train the network
        scores, queue = train(train_loader, model, optimizer, scaler, epoch, lr_schedule, queue)
        training_stats.update(scores)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                # save_dict["amp"] = apex.amp.state_dict()
                save_dict["scaler"] = scaler.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if queue is not None:
            torch.save({"queue": queue}, queue_path)


def train(train_loader, model, optimizer, scaler, epoch, lr_schedule, queue):
    
    def _crop_stats(t):
        # t: (B,C,H,W) on GPU
        t_cpu = t.detach().float().cpu()
        t_safe = torch.nan_to_num(t_cpu, nan=0.0, posinf=0.0, neginf=0.0)
        finite_ratio = torch.isfinite(t_cpu).float().mean().item()
        return {
            "shape": tuple(t_cpu.shape),
            "dtype": str(t.dtype),
            "finite_ratio": finite_ratio,
            "min": float(t_safe.min().item()),
            "max": float(t_safe.max().item()),
            "mean": float(t_safe.mean().item()),
            "std": float(t_safe.std().item()),
        }
    
    def _find_suspect_samples(t, absmax_thr=50.0, std_thr=1e-6):
        # 返回：bad_nonfinite_idx, extreme_idx, constant_idx
        B = t.shape[0]
        t_cpu = t.detach().float().cpu()
        finite_per_sample = torch.isfinite(t_cpu).view(B, -1).all(dim=1)
        bad_nonfinite_idx = (~finite_per_sample).nonzero(as_tuple=True)[0].tolist()
    
        t_safe = torch.nan_to_num(t_cpu, nan=0.0, posinf=0.0, neginf=0.0)
        absmax = t_safe.view(B, -1).abs().max(dim=1).values
        extreme_idx = (absmax > absmax_thr).nonzero(as_tuple=True)[0].tolist()
    
        std = t_safe.view(B, -1).std(dim=1)
        constant_idx = (std < std_thr).nonzero(as_tuple=True)[0].tolist()
    
        return bad_nonfinite_idx, extreme_idx, constant_idx, absmax, std

    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        for ci, t in enumerate(inputs):
            if not torch.isfinite(t).all():
                t_cpu = t.detach().float().cpu()
                t_safe = torch.nan_to_num(t_cpu, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"[BAD INPUT] iter={iteration} crop={ci} shape={tuple(t.shape)}", flush=True)
                print(f"[BAD INPUT PATCH] sample0 Cx8x8=\n{t_safe[0, :, :8, :8]}", flush=True)
                raise RuntimeError("Non-finite in input crop")
        
        # --- optional: dump specific iters (只做你想看的) ---
        if args.rank == 0 and iteration in {6, 7, 8}:
            torch.save(
                {"iter": iteration, "inputs": [t.detach().cpu() for t in inputs]},
                os.path.join(args.dump_path, f"debug_{iteration}_inputs_rank{args.rank}.pth"),
            )

        # ===== input sanity (rank0 only) =====
        if args.rank == 0:
            for ci, t in enumerate(inputs):
                bad_nf, extreme_idx, constant_idx, absmax, std = _find_suspect_samples(t, absmax_thr=50.0)
        
                # 1) 真正 NaN/Inf（你现在已有）
                if bad_nf:
                    print(f"[BAD INPUT NONFINITE] iter={iteration} crop={ci} bad_idx={bad_nf} shape={tuple(t.shape)}", flush=True)
                    # 打印第一个坏样本的 patch
                    bi = bad_nf[0]
                    t_cpu = torch.nan_to_num(t.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
                    print(f"[BAD INPUT PATCH] iter={iteration} crop={ci} sample={bi} Cx8x8=\n{t_cpu[bi, :, :8, :8]}", flush=True)
                    torch.save({"iter": iteration, "crop": ci, "bad_idx": bad_nf, "tensor": t.detach().cpu()},
                               os.path.join(args.dump_path, f"bad_input_{iteration}_crop{ci}.pth"))
                    raise RuntimeError("Non-finite in input crop")
        
                # 2) 没有 NaN/Inf，但数值极端/常数（这才是你现在最可能的原因）
                if extreme_idx or constant_idx:
                    st = _crop_stats(t)
                    print(f"[SUSPECT INPUT] iter={iteration} crop={ci} stats={st}", flush=True)
        
                    if extreme_idx:
                        top = sorted([(i, float(absmax[i])) for i in extreme_idx], key=lambda x: -x[1])[:5]
                        print(f"  [EXTREME] absmax>50 samples top5={top}", flush=True)
                        bi = top[0][0]
                        t_cpu = torch.nan_to_num(t.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
                        print(f"  [EXTREME PATCH] crop={ci} sample={bi} Cx8x8=\n{t_cpu[bi, :, :8, :8]}", flush=True)
        
                    if constant_idx:
                        print(f"  [CONSTANT] std<1e-6 samples={constant_idx[:20]} (show first 20)", flush=True)

    
                
        # ============ multi-res forward passes (autocast for memory/speed) ... ============
        with autocast(enabled=args.use_fp16):
            # embedding, output = model(inputs)
            # embedding = embedding.detach()
            embedding, output, dbg = model(inputs, return_debug=True)
            bs = inputs[0].size(0)

        emb = embedding.detach().float()

        if args.rank == 0 and iteration in {6, 7, 8}:
            def _isfinite(t):
                return bool(torch.isfinite(t).all().item())
            print("[FWD CHECK]",
                  "iter", iteration,
                  "backbone_flat", _isfinite(dbg["backbone_flat"]),
                  "emb_pre_norm", _isfinite(dbg["emb_pre_norm"]),
                  "embedding", _isfinite(dbg["embedding"]),
                  "logits", _isfinite(dbg["logits"]),
                  flush=True)

        
            # 看每个样本的 L2 norm（如果你模型里做了 normalize，这个最关键）
            norms = torch.linalg.norm(torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0), dim=1)
            print(f"[EMB NORM] iter={iteration} norm min={norms.min().item():.6f} max={norms.max().item():.6f} mean={norms.mean().item():.6f}",
                  flush=True)

        # 只在异常时触发，不会刷屏
        try:
            assert_finite(output, "output", iteration, dump_dir=args.dump_path)
            assert_finite(embedding, "embedding", iteration, dump_dir=args.dump_path)
        except RuntimeError:
            if args.rank == 0:
                torch.save(
                    {"iter": iteration, "inputs": [x.detach().cpu() for x in inputs]},
                    os.path.join(args.dump_path, f"debug_{iteration}_inputs.pth"),
                )
                print(f"[DEBUG] dumped inputs to debug_{iteration}_inputs.pth", flush=True)
        
                # 这里直接打印“这批 inputs 的 stats + 一个 patch”
                for ci, t in enumerate(inputs):
                    st = _crop_stats(t)
                    print(f"[BAD FWD INPUT STATS] iter={iteration} crop={ci} stats={st}", flush=True)
                    t_cpu = torch.nan_to_num(t.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
                    print(f"[BAD FWD INPUT PATCH] iter={iteration} crop={ci} sample0 Cx8x8=\n{t_cpu[0, :, :8, :8]}", flush=True)
            raise


        # ============ swav loss (assignment in fp32 for stability) ... ============
        loss = 0.0
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                # out: logits for the crops used for assignment
                out = output[bs * crop_id: bs * (crop_id + 1)].detach().float()  # <- fp32
                assert_finite(out, f"out_crop{crop_id}", iteration, dump_dir=args.dump_path)

                # time to use the queue
                if queue is not None:
                    # queue is fp32 (we enforce it below), prototypes weights are fp32
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        queued_logits = torch.mm(queue[i], model.module.prototypes.weight.t()).float()
                        assert_finite(model.module.prototypes.weight, "prototypes.weight", iteration, dump_dir=args.dump_path)
                        assert_finite(queue[i], f"queue_feat_i{i}", iteration, dump_dir=args.dump_path)
                        assert_finite(queued_logits, f"queued_logits_i{i}", iteration, dump_dir=args.dump_path)
                        out = torch.cat((queued_logits, out), dim=0)

                    # fill the queue with fp32 embeddings
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs].detach().float()

                # get assignments (fp32 sinkhorn)
                q = distributed_sinkhorn(out)[-bs:]  # fp32
                assert_finite(q, f"q_crop{crop_id}", iteration, dump_dir=args.dump_path)

            # cluster assignment prediction
            subloss = 0.0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                # IMPORTANT: log_softmax in fp32 to prevent fp16 underflow/NaN
                x = (output[bs * v: bs * (v + 1)] / args.temperature).float()
                if not torch.isfinite(x).all():
                    print("[BAD X] iter", iteration, "crop_id", crop_id, "v", v, flush=True)
                    raise RuntimeError("x is non-finite")
                x = x.clamp(min=-50, max=50)  
                logp = F.log_softmax(x, dim=1)
                if not torch.isfinite(logp).all():
                    print("[BAD LOGP] iter", iteration, "crop_id", crop_id, "v", v, flush=True)
                    raise RuntimeError("logp is non-finite")
                assert_finite(logp, f"logp_crop{crop_id}_v{v}", iteration, dump_dir=args.dump_path)
                subloss -= torch.mean(torch.sum(q * logp, dim=1))
            loss += subloss / (np.sum(args.nmb_crops) - 1)

        loss /= len(args.crops_for_assign)
        if args.rank == 0 and iteration == 0:
            xo = (output.float() / args.temperature)
            print("[DEBUG] x=output/temp stats:",
                  "min", xo.min().item(),
                  "max", xo.max().item(),
                  "mean", xo.mean().item(),
                  "std", xo.std().item(),
                  "temp", args.temperature,
                  "lr", optimizer.param_groups[0]["lr"],
                  "scale", scaler.get_scale(),
                  flush=True)

        # after loss computed
        if not torch.isfinite(loss).all():
            print("[BAD LOSS] iter", iteration, "loss", loss, flush=True)
            raise RuntimeError("loss is non-finite before backward")

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            # 先缩放 loss，反向传播
            scaler.scale(loss).backward()
            # 反缩放梯度，之后才可以安全地手动修改 p.grad
            scaler.unscale_(optimizer)

            # 检查梯度是否先坏
            found_bad_grad = False
            bad_name = None
            for n, p in model.named_parameters():
                if p.grad is not None and (not torch.isfinite(p.grad).all()):
                    found_bad_grad = True
                    bad_name = n
                    break

            if found_bad_grad:
                if args.rank == 0:
                    print(f"[AMP OVERFLOW] iter={iteration} bad_grad={bad_name} scale={scaler.get_scale()} lr={optimizer.param_groups[0]['lr']}", flush=True)
                    try:
                        print("[FIRST BAD GRAD SEEN AT]", bad_grad_first["name"], flush=True)
                    except Exception:
                        pass
                        
                optimizer.zero_grad(set_to_none=True)
                scaler.update()   # 降低 loss scale
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
            # 冻结 prototypes 的梯度
            if iteration < args.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                    if "prototypes" in name and p.grad is not None:
                        p.grad = None
            # 再做一步带缩放的 step + update 缩放因子
            scaler.step(optimizer)
            scaler.update()
            for n, p in model.named_parameters():
                if not torch.isfinite(p.data).all():
                    print(f"[BAD PARAM] {n} at iter {iteration}", flush=True)
                    raise RuntimeError(f"Non-finite param: {n} at iter {iteration}")
        
        else:
            loss.backward()
            if iteration < args.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                   if "prototypes" in name and p.grad is not None:
                        p.grad = None
            optimizer.step()
            for n, p in model.named_parameters():
                if not torch.isfinite(p.data).all():
                    print(f"[BAD PARAM] {n} at iter {iteration}", flush=True)
                    raise RuntimeError(f"Non-finite param: {n} at iter {iteration}")

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    # lr=optimizer.optim.param_groups[0]["lr"],
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg), queue


@torch.no_grad()
def distributed_sinkhorn(out):
    # ---- enforce fp32 for numerical stability ----
    out = out.float()

    # scale by epsilon and apply max-shift to avoid exp overflow
    out = out / args.epsilon
    out = out - out.max()

    Q = torch.exp(out).t()  # Q is K-by-B
    B = Q.shape[1] * args.world_size  # number of samples to assign
    K = Q.shape[0]  # number of prototypes

    # make the matrix sum to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= (sum_Q + 1e-12)

    for _ in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= (sum_of_rows + 1e-12)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
        Q /= (sum_of_cols + 1e-12)
        Q /= B

    Q *= B
    return Q.t()

if __name__ == "__main__":
    main()
