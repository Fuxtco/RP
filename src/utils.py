# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import pickle
import os

import re

import numpy as np
import torch

from .logger import create_logger, PD_Stats

import torch.distributed as dist

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

logger = getLogger()

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def init_distributed_mode(args):
    """
    Initialize:
      - args.rank
      - args.world_size
      - args.gpu_to_work_on
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    elif args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])

        if "SLURM_NTASKS" in os.environ:
            args.world_size = int(os.environ["SLURM_NTASKS"])
        else:
            tasks_per_node_str = os.environ.get("SLURM_TASKS_PER_NODE", "1")
            tasks_per_node = int(re.match(r"\d+", tasks_per_node_str).group())
            args.world_size = int(os.environ.get("SLURM_NNODES", "1")) * tasks_per_node

        local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    else:
        # single process fallback
        args.rank = 0
        args.world_size = 1
        local_rank = 0

    # prepare distributed (only if actually multi-process)
    if args.world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
        args.gpu_to_work_on = local_rank % max(1, ngpu)
        torch.cuda.set_device(args.gpu_to_work_on)
    else:
        args.gpu_to_work_on = -1

def initialize_exp(params, *args, dump_params=True):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    # dump parameters
    if dump_params:
        pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create repo to store checkpoints
    params.dump_checkpoints = os.path.join(params.dump_path, "checkpoints")
    if not params.rank and not os.path.isdir(params.dump_checkpoints):
        os.mkdir(params.dump_checkpoints)

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.dump_path, "stats" + str(params.rank) + ".pkl"), args
    )

    # create a logger
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"), rank=params.rank
    )
    logger.info("============ Initialized logger ============")
    logger.info(f"[dist-ok] rank={params.rank} world_size={params.world_size} gpu={params.gpu_to_work_on}")

    if params.rank == 0:
        logger.info(
            "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(vars(params).items()))
        )

    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("")
    return logger, training_stats


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at {}".format(ckp_path))

    #============================== safe map_location for both DDP and single-process runs=======================
    if torch.cuda.is_available():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gpu_id = torch.distributed.get_rank() % torch.cuda.device_count()
        else:
            gpu_id = 0
        map_loc = "cuda:" + str(gpu_id)
    else:
        map_loc = "cpu"

    checkpoint = torch.load(ckp_path, map_location=map_loc)

    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # reload variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            k_eff = min(k, maxk)
            correct_k = correct[:k_eff].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
