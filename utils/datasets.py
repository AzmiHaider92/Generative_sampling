# utils_torch/datasets.py
import os, random, shutil, zipfile
from typing import Iterator, Tuple
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.datasets.utils import download_url

# ----------------- transforms -----------------
class CenterSquare:
    def __call__(self, img):
        s = min(img.size)  # PIL.Image.size -> (W,H)
        return TF.center_crop(img, s)

def _to_minus1_1():
    return transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5))  # after ToTensor()

def _build_transform(image_size=256, train=False):
    ops = [
        #CenterSquare(),
        #transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),     # [0,1], CHW
        _to_minus1_1(),            # [-1,1], CHW
    ]
    # If you want aug: ops.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(ops)

def _auto_workers():
    try:
        import psutil
        return max(4, min(8 * torch.cuda.device_count(), psutil.cpu_count(logical=True)))
    except Exception:
        return max(4, 4 * torch.cuda.device_count())

def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

# ----------------- Tiny-ImageNet helpers -----------------
TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def _prepare_tiny_imagenet(root_dir: str):
    """
    Downloads/extracts Tiny-ImageNet-200 into root_dir if missing,
    and rearranges val/ into class subfolders.
    Final layout:
      root_dir/
        tiny-imagenet-200/
          train/ n*/images/*.JPEG
          val/   n*/images/*.JPEG
    """
    os.makedirs(root_dir, exist_ok=True)
    tgt = os.path.join(root_dir, "tiny-imagenet-200")
    if not os.path.isdir(tgt):
        zip_path = os.path.join(root_dir, "tiny-imagenet-200.zip")
        if not os.path.isfile(zip_path):
            print(f"[tiny-imagenet] downloading to {zip_path} ...")
            download_url(TINY_URL, root_dir, filename="tiny-imagenet-200.zip", md5=None)
        print(f"[tiny-imagenet] extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root_dir)

    # reorganize val/ into class folders if needed
    val_dir = os.path.join(tgt, "val")
    images_dir = os.path.join(val_dir, "images")
    anno_file = os.path.join(val_dir, "val_annotations.txt")
    if os.path.isdir(images_dir) and os.path.isfile(anno_file):
        print("[tiny-imagenet] reorganizing val/ ...")
        # read mappings: filename -> class
        mapping = {}
        with open(anno_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
        # move images into class subdirs
        for fn in os.listdir(images_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            cls = mapping.get(fn, None)
            if cls is None:
                continue
            cls_dir = os.path.join(val_dir, cls, "images")
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(images_dir, fn)
            dst = os.path.join(cls_dir, fn)
            if not os.path.isfile(dst):
                shutil.move(src, dst)
        # remove original flat images dir and annotations
        try:
            shutil.rmtree(images_dir)
        except Exception:
            pass
        try:
            os.remove(anno_file)
        except Exception:
            pass
    return tgt

# ----------------- dataset builders -----------------
def _make_imagefolder_loader(root: str, split: str, batch_size: int,
                             world: int, rank: int, image_size=256, debug_overfit=0) -> DataLoader:
    is_train = (split == "train")
    tfm = _build_transform(image_size=image_size, train=is_train)
    ds = datasets.ImageFolder(root=root, transform=tfm)
    if debug_overfit:
        ds = torch.utils.data.Subset(ds, list(range(min(debug_overfit, len(ds)))))

    sampler = DistributedSampler(ds, shuffle=is_train, drop_last=True) if world > 1 else None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None) and is_train,
        sampler=sampler,
        num_workers=2, # _auto_workers(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        worker_init_fn=_seed_worker,
        #pin_memory_device="cuda" if torch.cuda.is_available() else ""
    )
    return loader

def _get_tiny_imagenet_iter(per_rank_bs: int, train: bool, debug_overfit: int):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank  = int(os.environ.get("RANK", "0"))

    print(f"[rank {rank}] per_rank_bs={per_rank_bs}  global_bs={per_rank_bs * world}")

    root_dir = os.environ.get("TINY_IMAGENET_ROOT", "./data")
    # ideally do this only on rank 0 and barrier; shown earlier
    base = _prepare_tiny_imagenet(root_dir)

    split = "train" if train else "val"
    split_dir = os.path.join(base, split)

    loader = _make_imagefolder_loader(
        split_dir, split, per_rank_bs, world, rank,
        image_size=64, debug_overfit=debug_overfit
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _iter():
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(torch.randint(0, 2**31-1, (1,)).item())
        for x_chw, y in loader:
            x_chw = x_chw.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            yield x_chw.permute(0, 2, 3, 1).contiguous(), y  # BHWC
    return _iter()

def _get_cifar100_iter(batch_size: int, train: bool, debug_overfit: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    per_rank_bs = max(1, batch_size // world)

    tfm = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        _to_minus1_1(),
    ])
    ds = datasets.CIFAR100(root="./data", train=train, download=True, transform=tfm)
    if debug_overfit:
        ds = torch.utils.data.Subset(ds, list(range(min(debug_overfit, len(ds)))))

    sampler = DistributedSampler(ds, shuffle=train, drop_last=True) if world > 1 else None
    loader = DataLoader(
        ds,
        batch_size=per_rank_bs,
        shuffle=(sampler is None) and train,
        sampler=sampler,
        num_workers=_auto_workers(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
        worker_init_fn=_seed_worker,
        pin_memory_device="cuda" if torch.cuda.is_available() else ""
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _iter():
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(torch.randint(0, 2**31-1, (1,)).item())
        for x_chw, y in loader:
            x_chw = x_chw.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            yield x_chw.permute(0, 2, 3, 1).contiguous(), y
    return _iter()

# ----------------- public API -----------------
def get_dataset(dataset_name: str, batch_size: int, train: bool, debug_overfit: int = 0
               ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns (images_bhwc, labels) on GPU, float32 in [-1,1], BHWC.
    Supported dataset_name:
      - "tiny-imagenet-256"  -> auto-downloads Tiny-ImageNet-200, centersquare->256
      - "cifar100-256"       -> downloads CIFAR-100, upscales to 256
    """
    name = dataset_name.lower()
    if name.startswith("tiny-imagenet"):
        return _get_tiny_imagenet_iter(batch_size, train, debug_overfit)
    elif name.startswith("cifar100"):
        return _get_cifar100_iter(batch_size, train, debug_overfit)
    else:
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. "
            f"Use 'tiny-imagenet-256' or 'cifar100-256' (or plug your own ImageFolder)."
        )
