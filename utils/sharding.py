# utils_torch/sharding.py
import os
import torch
from datetime import timedelta
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=1800))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 1

def make_samplers(train_dataset, is_ddp: bool, shuffle=True):
    if is_ddp:
        return DistributedSampler(train_dataset, shuffle=shuffle)
    return None
