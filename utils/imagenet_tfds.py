import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode as IM
import io, os
from PIL import Image
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import time
# pip install tfrecord
from tfrecord.torch.dataset import TFRecordDataset
try:
    from tfrecord.tools.tfrecord2idx import create_index as _create_index
except Exception:
    _create_index = None


def _build_transform(is_train: bool):
    # pick output size
    target = 256

    # crop to square of min(H,W), then resize -> (target, target)
    def center_crop_square_then_resize(img):
        # Works for PIL.Image or torch.Tensor
        if isinstance(img, torch.Tensor):
            # Tensor could be CHW or HWC
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):  # CHW
                _, h, w = img.shape
                s = min(h, w)
                top = (h - s) // 2
                left = (w - s) // 2
                img = img[:, top:top+s, left:left+s]
                img = TF.resize(img, [target, target], interpolation=IM.BICUBIC, antialias=True)
                return img
            elif img.ndim == 3:  # HWC
                h, w, _ = img.shape
                s = min(h, w)
                top = (h - s) // 2
                left = (w - s) // 2
                img = img[top:top+s, left:left+s, :]
                img = TF.resize(img.permute(2, 0, 1), [target, target], interpolation=IM.BICUBIC, antialias=True)
                return img
            else:
                raise ValueError("Unexpected tensor shape for image.")
        else:
            # PIL.Image: size = (W, H)
            s = min(img.size)
            img = TF.center_crop(img, s)
            img = TF.resize(img, (target, target), interpolation=IM.BICUBIC, antialias=True)
            return img

    tfms = [T.Lambda(center_crop_square_then_resize)]
    if is_train:
        tfms.append(T.RandomHorizontalFlip())
    tfms += [
        T.ToTensor(),                       # [0,1]
        T.Normalize([0.5, 0.5, 0.5],        # -> [-1,1]
                    [0.5, 0.5, 0.5]),
    ]
    return T.Compose(tfms)


def _auto_workers(world: int) -> int:
    try:
        import psutil; cpus = psutil.cpu_count(logical=True) or (os.cpu_count() or 8)
    except Exception:
        cpus = os.cpu_count() or 8
    per_rank = max(2, (cpus // max(world, 1)) - 2)
    return min(per_rank, 8)


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception:
        pass


def _identity(x):  # top-level = picklable
    return x


def _find_shards(direc: str, split: str):
    pat = f"imagenet2012-{split}.tfrecord-"
    return sorted(
        os.path.join(direc, f)
        for f in os.listdir(direc)
        if f.startswith(pat) and not f.endswith(".index") and os.path.isfile(os.path.join(direc, f))
    )


def _ensure_index(tfr_path: str, index_root: str) -> str:
    os.makedirs(index_root, exist_ok=True)
    idx_path = os.path.join(index_root, os.path.basename(tfr_path) + ".index")
    if not os.path.isfile(idx_path):
        _create_index(tfr_path, idx_path)   # do on rank 0 only in multi-GPU
    return idx_path


def _ensure_all_indexes(shards, index_root: str, world: int = 1, rank: int = 0):
    os.makedirs(index_root, exist_ok=True)
    idxs = [os.path.join(index_root, os.path.basename(p) + ".index") for p in shards]

    if world > 1:
        if rank == 0:
            for p in shards:
                _ensure_index(p, index_root)        # creates only if missing
        else:
            for ip in idxs:                         # wait for rank 0 to write them
                while not os.path.isfile(ip):
                    time.sleep(0.1)
    else:
        for p in shards:
            _ensure_index(p, index_root)

    return idxs


# optional: useful for __len__
def _index_count(idx_path: str) -> int:
    with open(idx_path, "rb") as f:
        return sum(1 for _ in f)


# ---- IterableDataset wrapper ----
# choose keys for *your* shards
_IMG_KEY   = "image"              # was "image/encoded"
_LABEL_KEY = "label"              # was "image/class/label"


def _build_desc(img_key=_IMG_KEY, label_key=_LABEL_KEY):
    return {img_key: "byte", label_key: "int"}


class ImageNetTFRecord(IterableDataset):
    def __init__(self, ds_root: str, train: bool, world: int, rank: int, image_size: int = 256):
        self.world, self.rank = max(1, world), rank
        split = "train" if train else "validation"
        self.shards  = _find_shards(ds_root, split)
        index_root = r'./data/imagenet_index_dir'
        self.indexes = _ensure_all_indexes(self.shards, index_root, self.world, self.rank)
        # use your schema
        self.img_key, self.label_key = _IMG_KEY, _LABEL_KEY
        self.desc    = _build_desc(self.img_key, self.label_key)
        self.tfm     = _build_transform(train)
        # optional length estimate if you want
        self._size_rank = sum(_index_count(idx) for idx in self.indexes[self.rank::self.world])


    def __len__(self):  # optional; okay for IterableDataset
        return self._size_rank

    def _split_for_worker(self):
        wi = get_worker_info()
        nworkers = wi.num_workers if wi else 1
        wid      = wi.id if wi else 0
        # split shards by rank, then by worker
        s_rank = self.shards[self.rank::self.world]
        i_rank = self.indexes[self.rank::self.world]
        return s_rank[wid::nworkers], i_rank[wid::nworkers]

    def __iter__(self):
        shards_w, idxs_w = self._split_for_worker()
        if not shards_w:
            return
        for data_p, idx_p in zip(shards_w, idxs_w):
            ds = TFRecordDataset(data_p, idx_p, self.desc, transform=_identity, shuffle_queue_size=0)
            for sample in ds:
                img_bytes = sample[self.img_key]   # <-- 'image'
                with Image.open(io.BytesIO(img_bytes)) as im:
                    im = im.convert("RGB")
                x_chw = self.tfm(im).contiguous()

                label = int(sample[self.label_key])   # <-- 'label'
                # only subtract 1 if your labels are 1..1000; otherwise keep as-is
                # if 1 <= label <= 1000: label -= 1

                yield x_chw, torch.tensor(label, dtype=torch.long)


def get_imagenet_tfrecord_iter(ds_root, per_rank_bs, train, world, rank, image_size=256, debug_overfit=0):
    ds = ImageNetTFRecord(ds_root, train, world, rank, image_size=image_size)

    # Windows uses spawn ⇒ start with num_workers=0; on Linux you can bump it
    num_workers = 2 # 0 if os.name == "nt" else _auto_workers(world)

    loader = DataLoader(
        ds,
        batch_size=per_rank_bs,
        shuffle=False,                 # IterableDataset: do shuffling inside if needed
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        drop_last=True,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def _iter():
        epoch = 0
        while True:
            for x_chw, y in loader:
                yield x_chw.to(device, non_blocking=False).permute(0,2,3,1).contiguous(), y.to(device, non_blocking=False)
            epoch +=1
    return _iter()


def _get_imagenet_iter(dataset_root_dir: str, batch_size: int, train: bool, debug_overfit: int, image_size: int = 256):
    """
    ImageNet (TFRecord shards) → iterator yielding (images_bhwc, labels) on GPU.
    Expects shards like: imagenet2012-{train|val}.tfrecord-00000-of-01024
    Set IMAGENET_TFRECORD_ROOT to the directory containing those files.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank  = int(os.environ.get("RANK", "0"))

    #root = os.environ.get("IMAGENET_TFRECORD_ROOT", "./data/imagenet-tfrecords")
    try:
        return get_imagenet_tfrecord_iter(
            ds_root=dataset_root_dir,
            per_rank_bs=batch_size,   # <-- this is already per-GPU
            train=train,
            world=world,
            rank=rank,
            debug_overfit=debug_overfit,
            image_size=image_size
        )
    except FileNotFoundError as e:
        raise ValueError(
            f"Could not find ImageNet TFRecord shards under '{dataset_root_dir}'. "
            f"Set IMAGENET_TFRECORD_ROOT to the folder with files like "
            f"'imagenet2012-train.tfrecord-00000-of-01024'."
        ) from e
