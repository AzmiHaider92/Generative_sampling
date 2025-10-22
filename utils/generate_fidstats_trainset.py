# make_real_fid_stats.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from utils.celeba_hq import FlatImageFolderSafe
from utils.datasets import get_dataset as get_dataset_iter
from utils.imagenet_tfds import ImageNetTFRecord


# ----------------------------
# 1) Your dataloader goes here
# ----------------------------
# Replace this with your actual dataset / loader. It must yield tensors shaped [B,3,H,W]
# Example:
# real_loader = DataLoader(MyImagenetTFRecordsDataset(...), batch_size=128, num_workers=8, pin_memory=True)


def make_real_stats(
    loader,
    stats_out_path: str = "imagenet256_fidstats.pt",
    device: str = "cuda",
    image_size: int = 256,
):
    """
    Streams real images from `real_loader`, computes FID 'real' stats, and saves TorchMetrics state_dict to disk.
    The saved file contains everything needed to later compute FID against generated images.
    """
    os.makedirs(os.path.dirname(stats_out_path) or ".", exist_ok=True)

    # TorchMetrics FID expects 3xHxW images. We’ll feed normalized in [0,1] and let it handle internal resize to 299.
    fid = FrechetInceptionDistance(
        feature=2048,
        normalize=True,            # interpret inputs as floats in [0,1]
        input_img_size=(3, image_size, image_size),
        antialias=True,
    ).to(device)

    imgs_seen = 0
    pbar = tqdm(total=None, unit="img")  # no fixed total; count images instead
    fid.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[0]

            images = images.to(device, non_blocking=True)

            # If your pipeline yields [-1,1], convert to [0,1]
            images = (images + 1.0) * 0.5
            images = images.clamp(0.0, 1.0)
            #images = images.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW

            # Update real stats
            fid.update(images, real=True)
            imgs_seen += images.shape[0]
            pbar.update(images.shape[0])

    # Save only the metric state (compact: sums, counts, etc.), not raw features
    stats = {
            "real_features_sum":         fid.real_features_sum.detach().cpu(),
            "real_features_cov_sum":     fid.real_features_cov_sum.detach().cpu(),
            "real_features_num_samples": fid.real_features_num_samples.detach().cpu(),
            "feature_dim":               2048,  # for safety
            }
    torch.save(stats, stats_out_path)
    print(f"Saved real FID stats to: {stats_out_path}")
    return imgs_seen


if __name__ == "__main__":
    ds_root = r'C:\Users\azmih\Desktop\Projects\datasets\celebA_hq_256\celeba_hq_256'
    batch_size = 32
    image_size = 256
    #ds = ImageNetTFRecord(ds_root, True, 1, 0, image_size=256)
    ds = FlatImageFolderSafe(root=ds_root, image_size=image_size)

    # Windows uses spawn ⇒ start with num_workers=0; on Linux you can bump it
    num_workers = 0 if os.name == "nt" else 1

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # IterableDataset: do shuffling inside if needed
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        drop_last=False,
        worker_init_fn=None,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_images = make_real_stats(loader,
    stats_out_path=r"C:\Users\azmih\Desktop\Projects\Generative_sampling\data\celeba256_fidstats_gt.pt",
    device=device,
    image_size=image_size)
    print(f"Number of images seen in fid calc: {num_images}")
