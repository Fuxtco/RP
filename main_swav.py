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

# apex
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")


parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")


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
    elif args.sync_bn == "apex":
        raise ValueError("Apex SyncBN is not available. Please use --sync_bn pytorch or none.")
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # FP32
    if model.prototypes is not None:
        model.prototypes = model.prototypes.to(dtype=torch.float32)

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=True)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision (torch.cuda.amp)
    scaler = GradScaler(enabled=args.use_fp16)
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
                
        '''
        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)
        '''
        # normalize the prototypes (with safety checks)
        if model.module.prototypes is not None:
            with torch.no_grad():
                w0 = model.module.prototypes.weight.data
                if not torch.isfinite(w0).all():
                    n_nan = torch.isnan(w0).sum().item()
                    n_inf = torch.isinf(w0).sum().item()
                    logger.error(
                        f"[rank={args.rank}] Non-finite prototypes weight before norm "
                        f"at epoch={epoch}, it={it}: nan={n_nan}, inf={n_inf}"
                    )
                    raise RuntimeError("Non-finite prototypes weights, stopping training")

                w = F.normalize(w0, dim=1, p=2).detach()          # normalize in fp32
                model.module.prototypes.weight.copy_(w)           # keep original dtype


        # ---- debug: prototype dtype (only once) ----
        if args.rank == 0 and epoch == 0 and it == 0:
            w = model.module.prototypes.weight
            logger.info(
                f"[DEBUG] prototypes.weight dtype={w.dtype}, "
                f"min={w.min().item():.4g}, max={w.max().item():.4g}"
            )

        inputs = [x.cuda(non_blocking=True) for x in inputs]
        # ============ multi-res forward passes (autocast for memory/speed) ... ============
        with autocast(enabled=args.use_fp16):
            # Forward pass: obtain embeddings and prototype logits
            embedding, output = model(inputs)

            # ---- debug: forward dtype (only once) ----
            if args.rank == 0 and epoch == 0 and it == 0:
                logger.info(
                    f"[DEBUG] embedding dtype={embedding.dtype}, "
                    f"output dtype={output.dtype}"
                )

            # ---- Numerical stability check (model output) ----
            # If NaN or Inf appears in the model outputs (prototype logits),
            # the training is already numerically unstable and must be stopped
            # immediately to avoid log explosion and corrupted checkpoints.
            if not torch.isfinite(output).all():
                lr = optimizer.param_groups[0]["lr"]
                n_nan = torch.isnan(output).sum().item()
                n_inf = torch.isinf(output).sum().item()

                if args.rank == 0:
                    finite_mask = torch.isfinite(output)
                    if finite_mask.any():
                        o_min = output[finite_mask].min().item()
                        o_max = output[finite_mask].max().item()
                    else:
                        o_min, o_max = float("nan"), float("nan")

                    logger.error(
                        f"[rank={args.rank}] Non-finite output at epoch={epoch}, it={it}, lr={lr} | "
                        f"nan={n_nan}, inf={n_inf}, finite_min={o_min:.6g}, finite_max={o_max:.6g}, "
                        f"shape={tuple(output.shape)}, dtype={output.dtype}"
                    )
                else:
                    logger.error(
                        f"[rank={args.rank}] Non-finite output at epoch={epoch}, it={it}, lr={lr} | "
                        f"nan={n_nan}, inf={n_inf}"
                    )

                raise RuntimeError("Non-finite output, stopping training")
            # ---- Numerical stability check ----

            # Detach embeddings to avoid backpropagation through the assignment path
            embedding = embedding.detach()

            # ---- Numerical stability check (embedding) ----
            # Even if the output logits are finite, embeddings may already contain
            # NaN/Inf values (e.g. due to fp16 overflow or unstable gradients).
            # Detecting this early prevents silent corruption of the queue.
            if not torch.isfinite(embedding).all():
                if args.rank == 0:
                    logger.error(f"Non-finite embedding at epoch={epoch}, it={it}")
                raise RuntimeError("Non-finite embedding, stopping training")
            # ---- Numerical stability check ----

            # Batch size for one crop (used in SwAV loss computation)
            bs = inputs[0].size(0)

        # ============ swav loss (assignment in fp32 for stability) ... ============
        loss = 0.0
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                # out: logits for the crops used for assignment
                out = output[bs * crop_id: bs * (crop_id + 1)].detach().float()  # <- fp32

                # time to use the queue
                if queue is not None:
                    # queue is fp32 (we enforce it below), prototypes weights are fp32
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        queued_logits = torch.mm(queue[i], model.module.prototypes.weight.t()).float()
                        out = torch.cat((queued_logits, out), dim=0)

                    # fill the queue with fp32 embeddings
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs].detach().float()

                # get assignments (fp32 sinkhorn)
                q = distributed_sinkhorn(out)[-bs:]  # fp32

            # cluster assignment prediction
            subloss = 0.0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                # IMPORTANT: log_softmax in fp32 to prevent fp16 underflow/NaN
                x = (output[bs * v: bs * (v + 1)] / args.temperature).float()
                x = x.clamp(min=-50, max=50)  
                logp = F.log_softmax(x, dim=1)
                subloss -= torch.mean(torch.sum(q * logp, dim=1))
            loss += subloss / (np.sum(args.nmb_crops) - 1)
        loss /= len(args.crops_for_assign)

        # --- stop early on NaN/Inf to avoid huge logs / wasted GPU time ---
        if not torch.isfinite(loss):
            logger.error(
                f"NaN detected at epoch {epoch}, iter {it}, loss={loss.item()}"
            )
            raise RuntimeError("NaN loss, stopping training")

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)           
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # cancel gradients for prototypes during the freeze period (same behavior as original SwAV)
            if iteration < args.freeze_prototypes_niters and model.module.prototypes is not None:
                for p in model.module.prototypes.parameters():
                    if p.grad is not None and (not torch.isfinite(p.grad).all()) and args.rank == 0:
                        logger.warning(
                            f"[rank={args.rank}] Non-finite prototypes grad ignored (frozen) "
                            f"at epoch={epoch}, it={it}"
                        )
                    p.grad = None

            scaler.step(optimizer)
            scaler.update()
        
        else:
            loss.backward()
            if iteration < args.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                   if "prototypes" in name and p.grad is not None:
                        p.grad = None
            optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 100 == 0:
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

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


if __name__ == "__main__":
    main()
