# utils_torch/stable_vae.py
import torch
from diffusers import AutoencoderKL

# code edited to receive  BCHW (instead of BHWC)
class StableVAE:
    def __init__(self, device):
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        self.device = device
        self.scaling_factor = 0.18215
        self.vae.eval()

    @torch.no_grad()
    def encode(self, x_bchw: torch.Tensor) -> torch.Tensor:
        #x = x_bhwc.permute(0, 3, 1, 2).contiguous()
        latents = self.vae.encode(x_bchw).latent_dist.sample() * self.scaling_factor
        return latents

    @torch.no_grad()
    def decode(self, z_bchw: torch.Tensor) -> torch.Tensor:
        x = self.vae.decode(z_bchw / self.scaling_factor).sample
        #x = x.permute(0, 2, 3, 1).contiguous()
        return x.clamp(-1, 1)
