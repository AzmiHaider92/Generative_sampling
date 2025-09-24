# model(torch).py
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------- Positional encodings (2D sin-cos) -------
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = torch.einsum('m,d->md', pos.reshape(-1), omega)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb

def get_2d_sincos_pos_embed(embed_dim, length, device):
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)
    gh, gw = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack([gh.reshape(-1), gw.reshape(-1)], dim=0)  # [2, H*W]
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1)  # [H*W, D]
    return emb.unsqueeze(0)  # [1, H*W, D]

# ------- Small helpers -------
def modulate(x, shift, scale):
    # Match your JAX model.py version (no clipping inside modulate).
    # (You also had a clipped version in math_utils.py—this mirrors model.py.)
    # x: [B, L, C], shift/scale: [B, C]
    return x * (1 + scale[:, None, :]) + shift[:, None, :]

@dataclass
class TrainConfig:
    dtype: torch.dtype = torch.bfloat16
    def default_kwargs(self):
        return dict(dtype=self.dtype)

# ------- Building blocks -------
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, tc: TrainConfig, freq_size: int = 256):
        super().__init__()
        self.hidden = hidden_size
        self.freq = freq_size
        self.tc = tc
        self.fc1 = nn.Linear(self.freq, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, t):  # t in [0,1], shape [B]
        t = t.to(torch.float32)
        half = self.freq // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None] * freqs[None, :]

        # build emb in fp32, then cast to fc1’s weight dtype
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        emb = emb.to(self.fc1.weight.dtype)

        x = F.silu(self.fc1(emb))
        x = self.fc2(x)

        # (optional) cast output to tc.dtype if the rest of your net expects it
        if hasattr(self.tc, "dtype") and self.tc.dtype is not None and x.dtype != self.tc.dtype:
            x = x.to(self.tc.dtype)
        return x

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, tc: TrainConfig):
        super().__init__()
        self.emb = nn.Embedding(num_classes + 1, hidden_size)  # +1 for null token
        self.tc = tc
    def forward(self, labels):  # [B] int64
        return self.emb(labels)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, hidden_size: int, tc: TrainConfig, bias: bool = True):
        super().__init__()
        self.patch = patch_size
        self.tc = tc
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=bias
        )

    def forward(self, x_bhwc):
        B, H, W, C = x_bhwc.shape
        x = x_bhwc.permute(0,3,1,2).contiguous()
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        return x, (H // self.proj.kernel_size[0], W // self.proj.kernel_size[0])


class MlpBlock(nn.Module):
    def __init__(self, hidden: int, mlp_ratio: float, dropout: float, tc: TrainConfig):
        super().__init__()
        inner = int(hidden * mlp_ratio)
        self.fc1 = nn.Linear(hidden, inner)
        self.fc2 = nn.Linear(inner, hidden)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.drop(F.gelu(self.fc1(x), approximate="tanh"))
        x = self.drop(self.fc2(x))
        return x

class DiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float, dropout: float, tc: TrainConfig):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.tc = tc
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.dropout = dropout
        self.mlp = MlpBlock(hidden, mlp_ratio, dropout, tc)

        self.ada = nn.Linear(hidden, 6 * hidden)  # to produce shift/scale/gate for attn & mlp

    def forward(self, x, c):  # x:[B,L,C], c:[B,C]
        # adaLN-Zero conditioning
        c = F.silu(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(c).chunk(6, dim=-1)

        # Attention block
        x_norm = self.norm1(x)
        x_mod = modulate(x_norm, shift_msa, scale_msa)
        B, L, C = x_mod.shape
        H = self.heads
        q = self.q(x_mod).view(B, L, H, C // H).transpose(1, 2)  # [B,H,L,D]
        k = self.k(x_mod).view(B, L, H, C // H).transpose(1, 2)
        v = self.v(x_mod).view(B, L, H, C // H).transpose(1, 2)
        # Your JAX used q /= D; SDPA handles scaling internally; keep close to original:
        #q = q / (q.shape[-1])
        #attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)  # [B,H,L,L]
        #y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, C)
        y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=(self.dropout if self.training else 0.0)  # or define self.attn_drop in __init__
                ,is_causal=False
                )  # [B,H,L,D]

        y = y.transpose(1, 2).contiguous().view(B, L, C)

        y = self.proj(y)
        x = x + gate_msa[:, None, :] * y

        # MLP block
        x_norm2 = self.norm2(x)
        x_mod2 = modulate(x_norm2, shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(x_mod2)
        return x

class FinalLayer(nn.Module):
    def __init__(self, patch_size: int, out_channels: int, hidden: int, tc: TrainConfig):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.to_mod = nn.Linear(hidden, 2 * hidden)
        self.proj = nn.Linear(hidden, patch_size * patch_size * out_channels)
        self.patch = patch_size
        self.outc = out_channels
    def forward(self, x, c):
        c = F.silu(c)
        shift, scale = self.to_mod(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.proj(x)  # [B, L, P*P*C]
        return x

class DiT(nn.Module):
    """
    Torch port of your JAX DiT with adaLN-Zero and (t, dt, y) conditioning.
    API mirrors: forward(x, t, dt, y, train=False, return_activations=False)
    """
    def __init__(self,
                 in_channels: int, patch_size: int, hidden_size: int, depth: int, num_heads: int,
                 mlp_ratio: float, out_channels: int, class_dropout_prob: float,
                 num_classes: int, ignore_dt: bool = False, dropout: float = 0.0,
                 dtype: torch.dtype = torch.bfloat16, image_size: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.patch = patch_size
        self.hidden = hidden_size
        self.depth = depth
        self.heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.outc = out_channels
        self.class_dropout = class_dropout_prob
        self.num_classes = num_classes
        self.ignore_dt = ignore_dt
        self.dropout = dropout
        self.tc = TrainConfig(dtype=dtype)

        self.patch_embed = PatchEmbed(in_channels, patch_size, hidden_size, self.tc)
        self.te = TimestepEmbedder(hidden_size, self.tc)
        self.dte = TimestepEmbedder(hidden_size, self.tc)
        self.ye = LabelEmbedder(num_classes, hidden_size, self.tc)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout, self.tc)
            for _ in range(depth)
        ])
        self.final = FinalLayer(patch_size, out_channels, hidden_size, self.tc)

        # logvars lookup by discrete t (0..255), multiplied by 100 like your JAX
        #self.logvar_table = nn.Embedding(256, 1)
        #nn.init.constant_(self.logvar_table.weight, 0.0)

        self.image_size_hint = image_size  # optional; not strictly needed

    @torch.no_grad()
    def _get_pos_embed(self, num_patches, device):
        return get_2d_sincos_pos_embed(self.hidden, num_patches, device=device).to(self.tc.dtype)

    def forward(self, x_bhwc, t, dt, y, train: bool = False, return_activations: bool = False):
        # x: [B,H,W,C] in [-?], t:[B] in [0,1], dt:[B] (ignored if self.ignore_dt), y:[B] (int)
        B, H, W, C = x_bhwc.shape
        if self.ignore_dt:
            dt = torch.zeros_like(t)

        x_tok, (Hp, Wp) = self.patch_embed(x_bhwc)      # [B, L, C]
        L = Hp * Wp
        pos = self._get_pos_embed(L, x_tok.device)
        x = (x_tok + pos).to(self.tc.dtype)

        te = self.te(t)
        dte = self.dte(dt)
        ye = self.ye(y)
        c = te + dte + ye  # [B, hidden]

        activ = {}
        if return_activations:
            activ['patch_embed'] = x
            activ['pos_embed'] = pos
            activ['time_embed'] = te
            activ['dt_embed'] = dte
            activ['label_embed'] = ye
            activ['conditioning'] = c

        for i, blk in enumerate(self.blocks):
            x = blk(x, c)
            if return_activations:
                activ[f'dit_block_{i}'] = x

        out = self.final(x, c)  # [B, L, P*P*C]
        if return_activations:
            activ['final_layer'] = out

        # Fold tokens back to image
        out = out.view(B, Hp, Wp, self.patch, self.patch, self.outc)       # [B, Hp, Wp, p, p, C]
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()                    # [B, Hp, p, Wp, p, C]
        out = out.view(B, Hp * self.patch, Wp * self.patch, self.outc)      # [B, H, W, C]

        # discrete t for logvars (0..255)
        #t_disc = torch.clamp((t * 256).floor().to(torch.int64), 0, 255)
        logvars = 0 #100.0 * self.logvar_table(t_disc)  # [B,1]

        if return_activations:
            return out, logvars, activ
        return out, None, None
