import os, glob, random, warnings
from typing import Iterator, Tuple, List, Optional
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import transforms

# --- make PIL robust to truncated files ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# --- use filesystem sharing to avoid /dev/shm pressure ---
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception:
        pass


def _build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
    ])


class FlatImageFolderSafe(Dataset):
    """
    CelebA-style flat folder with robust __getitem__ that
    skips unreadable images instead of crashing a worker.
    """
    def __init__(self, root: str, image_size: int, debug_overfit: int = 0):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        files: List[str] = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f"No images found under: {root}")

        if debug_overfit and debug_overfit > 0:
            files = files[:debug_overfit]

        self.files = files
        self.image_size = image_size
        self.T = _build_transform(image_size)
        # optional quick filter pass (cheap) — drop zero-byte files
        self.files = [f for f in self.files if (os.path.getsize(f) > 0)]

    def __len__(self):
        return len(self.files)

    def _safe_load(self, path: str) -> Optional[torch.Tensor]:
        # return None on failure instead of raising
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
            return self.T(im)  # CHW in [-1,1]
        except Exception:
            return None

    def __getitem__(self, idx):
        # Try the requested index; if bad, try a few random fallbacks
        path = self.files[idx]
        x = self._safe_load(path)
        if x is None:
            for _ in range(3):
                alt = self.files[random.randrange(0, len(self.files))]
                x = self._safe_load(alt)
                if x is not None:
                    break
        if x is None:
            # As a last resort, return a dummy (keeps batch shapes aligned)
            x = torch.full((3, self.image_size, self.image_size), -1.0)
            #x = torch.zeros(3, self.T.transforms[1].size, self.T.transforms[1].size)
        y = 0
        return x, y

def _collate_drop_none(batch):
    # Our loader never returns None now, but if you add skipping logic later, this keeps things safe.
    batch = [(x, y) for (x, y) in batch if x is not None]
    xs = torch.stack([x for x, _ in batch], dim=0)
    ys = torch.tensor([y for _, y in batch], dtype=torch.long)
    return xs, ys


def _get_celeba_iter(root: str, per_rank_bs: int, train: bool, debug_overfit: int, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank  = int(os.environ.get("RANK", "0"))
    print(f"[rank {rank}] per_rank_bs={per_rank_bs}  global_bs={per_rank_bs * world}")

    dataset = FlatImageFolderSafe(root=root, image_size=image_size, debug_overfit=debug_overfit)

    if world > 1:
        # drop_last=True ensures identical iteration counts across ranks
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=train, drop_last=True)
    else:
        sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)

    # Conservative worker settings for flat tiny-file datasets
    num_workers = 0 if os.name == "nt" else 2
    loader = DataLoader(
        dataset,
        batch_size=per_rank_bs,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() and num_workers > 0),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        drop_last=True,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _iter():
        epoch = 0
        while True:
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

            for x_chw, y in loader:
                x_chw = x_chw.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).long()
                yield x_chw, y
            epoch += 1

    return _iter()
