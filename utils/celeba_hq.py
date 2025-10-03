import os, math, glob
from typing import Iterator, Tuple, List
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import transforms

# ---------- tiny dataset helper ----------
class FlatImageFolder(Dataset):
    def __init__(self, root: str, image_size: int, debug_overfit: int = 0):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        files: List[str] = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        if not files:
            raise FileNotFoundError(f"No images found under: {root}")

        if debug_overfit and debug_overfit > 0:
            files = files[:debug_overfit]

        self.files = files
        self.transform = transforms.Compose([
            transforms.ToTensor(),                            # [0,1]
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),  # -> [-1,1]
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
        x = self.transform(im)                     # CHW in [-1,1]
        y = 0                                      # dummy label (no classes)
        return x, y


# ---------- main iterator ----------
def _get_celeba_iter(root: str, per_rank_bs: int, train: bool, debug_overfit: int, image_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank  = int(os.environ.get("RANK", "0"))
    print(f"[rank {rank}] per_rank_bs={per_rank_bs}  global_bs={per_rank_bs * world}")

    dataset = FlatImageFolder(root=root, image_size=image_size, debug_overfit=debug_overfit)

    if world > 1:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=train, drop_last=False)
    else:
        sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=per_rank_bs,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(4 > 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _iter():
        # new epoch shuffling for DDP
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(torch.randint(0, 2**31 - 1, (1,)).item())

        for x_chw, y in loader:
            x_chw = x_chw.to(device, non_blocking=True)        # [B,C,H,W] in [-1,1]
            # y is dummy zeros; keep dtype long for code paths expecting class indices
            if not torch.is_tensor(y):
                y = torch.zeros(x_chw.size(0), dtype=torch.long)
            y = y.to(device, non_blocking=True).long()
            yield x_chw, y     # -> BHWC

    return _iter()
