import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor

@torch.no_grad()
def get_fid_network(device=None):

    # Use new weights API + keep aux_logits=True to match checkpoint
    weights = Inception_V3_Weights.DEFAULT  # or Inception_V3_Weights.IMAGENET1K_V1
    base = inception_v3(weights=weights, aux_logits=True, transform_input=False).to(device).eval()

    # Grab the global-average-pooled 2048-d features (“pool3” equivalent)
    feat = create_feature_extractor(base, return_nodes={'avgpool': 'pool'})

    def activations(x_bhwc: torch.Tensor) -> torch.Tensor:
        # x_bhwc is [-1,1]; map to [0,1] and resize to 299x299
        x = (x_bhwc + 1.0) / 2.0
        x = x.permute(0, 3, 1, 2).contiguous()                         # BHWC -> BCHW
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        out = feat(x)['pool']                                          # [N, 2048, 1, 1]
        return out.flatten(1)                                          # [N, 2048]

    return activations


def _stats(acts: np.ndarray):
    mu = acts.mean(axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma

def fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Standard FID formula
    diff = mu1 - mu2
    covmean, _ = np.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = np.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
