# utils_torch/sharding.py
import os
import torch
from datetime import timedelta
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def ddp_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    is_ddp = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))

    return is_ddp, device, rank, world_size, local_rank


def make_samplers(train_dataset, is_ddp: bool, shuffle=True):
    if is_ddp:
        return DistributedSampler(train_dataset, shuffle=shuffle)
    return None
