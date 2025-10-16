# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from typing import List, Tuple, Dict


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        num_classes += (num_classes > 1)
        self.embedding_table = nn.Embedding(num_classes, hidden_size) # + 1 for the non-class
        self.num_classes = num_classes

    #def token_drop(self, labels, force_drop_ids=None):
    #    """
    #    Drops labels to enable classifier-free guidance.
    #    """
    #    if force_drop_ids is None:
    #        drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
    #    else:
    #        drop_ids = force_drop_ids == 1
    #    labels = torch.where(drop_ids, self.num_classes, labels)
    #    return labels

    def forward(self, labels, train, force_drop_ids=None):
        #use_dropout = self.dropout_prob > 0
        #if (train and use_dropout) or (force_drop_ids is not None):
        #    labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# Assumes the following classes exist in your codebase and behave like DiT official:
# - PatchEmbed(image_size, patch_size, in_chans, embed_dim, bias=True)
# - TimestepEmbedder(hidden_size)
# - LabelEmbedder(num_classes, hidden_size)
# - DiTBlock(hidden_size, num_heads, mlp_ratio)
# - FinalLayer(hidden_size, patch_size, out_channels)
# - get_2d_sincos_pos_embed(D, grid_size) -> numpy array [T, D]


class WeightedBlendRefiner(nn.Module):
    """
    Produces per-sample mixture weights over K proposals using global pooled statistics
    and context embedding c. Returns v_ref = sum_k w_k * v_k and the weights.
    - proposals: (N, K, C, H, W)
    - v_main:    (N, C, H, W)
    - c:         (N, D) context embedding (t + y + k)
    """
    def __init__(self, c_dim: int, C: int, K: int, hidden: int = 256, proj_dim: int = 128):
        super().__init__()
        self.K = K
        self.c_proj = nn.Linear(c_dim, proj_dim)
        # Per-head MLP shared across heads; head id will be encoded implicitly by differences
        # in proposal feature vectors. Input per head = [gap(v_k), gap(|v_k - v_main|), c_proj]
        self.mlp = nn.Sequential(
            nn.Linear(2 * C + proj_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),  # scalar logit per head
        )

    def forward(self, proposals: torch.Tensor, v_main: torch.Tensor, c: torch.Tensor,
                temperature: float = 1.0, top1: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        N, K, C, H, W = proposals.shape
        assert K == self.K
        # Global average pool proposals and diffs
        gap = proposals.mean(dim=(3, 4))  # (N, K, C)
        diff_gap = (proposals - v_main.unsqueeze(1)).abs().mean(dim=(3, 4))  # (N, K, C)
        c_proj = self.c_proj(c)  # (N, proj_dim)
        c_rep = c_proj.unsqueeze(1).expand(N, K, -1)  # (N, K, proj_dim)
        feats = torch.cat([gap, diff_gap, c_rep], dim=-1)  # (N, K, 2C + proj_dim)
        logits = self.mlp(feats).squeeze(-1)  # (N, K)
        if temperature <= 0:
            temperature = 1.0
        if top1:
            # Straight argmax gating (non-differentiable choice). Still returns one-hot for clarity.
            idx = logits.argmax(dim=-1)  # (N,)
            w = torch.zeros_like(logits)
            w.scatter_(1, idx.unsqueeze(1), 1.0)
        else:
            w = torch.softmax(logits / temperature, dim=-1)  # (N, K)
        v_ref = (proposals * w.view(N, K, 1, 1, 1)).sum(dim=1)  # (N, C, H, W)
        return v_ref, w


class ProposalHead(nn.Module):
    """A lightweight head producing a velocity proposal from token features.
    Mirrors FinalLayer but keeps its own tiny parameters so heads can specialize.
    Input:  tokens x: (N, T, D), context c: (N, D)
    Output: (N, T, patch_size**2 * C)
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.final = FinalLayer(hidden_size, patch_size, out_channels)
        # Init like DiT: zero-out output layers for stable start
        nn.init.constant_(self.final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final.linear.weight, 0)
        nn.init.constant_(self.final.linear.bias, 0)

    def forward(self, x_tokens: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.final(x_tokens, c)


class DiT(nn.Module):
    """
    DiT with TRM-style multi-proposal heads and a learned refiner.

    Returns a dict:
      {
        'v_main': (N, C, H, W),
        'v_props': (N, K, C, H, W),
        'v_ref': (N, C, H, W),
        'weights': (N, K),
        'tokens': (N, T, D)   # optional, useful for debugging
      }
    """
    def __init__(self,
                 in_channels: int = 4,
                 patch_size: int = 2,
                 hidden_size: int = 1152,
                 depth: int = 28,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 num_classes: int = 1000,
                 ignore_k: bool = True,
                 image_size: int = 32,
                 num_of_proposals: int = 3,  # number of proposals
                 ref_hidden: int = 256,
                 ref_proj_dim: int = 128):
        super().__init__()
        assert num_of_proposals >= 1
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.ignore_k = ignore_k
        self.num_of_proposals = num_of_proposals

        # === Backbone (same as DiT) ===
        self.x_embedder = PatchEmbed(image_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.k_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # === Heads ===
        self.final_layer_main = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.proposal_heads = nn.ModuleList([
            ProposalHead(hidden_size, patch_size, self.out_channels) for _ in range(num_of_proposals)
        ])

        # === Refiner ===
        self.refiner = WeightedBlendRefiner(c_dim=hidden_size, C=self.out_channels, K=num_of_proposals,
                                            hidden=ref_hidden, proj_dim=ref_proj_dim)
        self.initialize_weights()

    # ---------- helpers ----------
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _context(self, t: torch.Tensor, k: torch.Tensor, y: torch.Tensor, train: bool) -> torch.Tensor:
        if self.ignore_k:
            k = torch.zeros_like(t)
        t_emb = self.t_embedder(t)
        k_emb = self.k_embedder(k)
        y_emb = self.y_embedder(y, train)
        return t_emb + y_emb + k_emb

    # ---------- init like DiT ----------
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # pos_embed (fixed sin/cos)
        grid_sz = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_sz)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # patch_embed like linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # label + timestep emb
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.k_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.k_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation in backbone blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out outputs (main head)
        nn.init.constant_(self.final_layer_main.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_main.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_main.linear.weight, 0)
        nn.init.constant_(self.final_layer_main.linear.bias, 0)
        # Proposal heads already zeroed in their ctor.

    # ---------- forward ----------
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                k: torch.Tensor,
                y: torch.Tensor,
                train: bool = True,
                temperature: float = 5.0,
                top1: bool = False,
                return_tokens: bool = False) -> Dict[str, torch.Tensor]:
        # tokens
        tokens = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        c = self._context(t, k, y, train)             # (N, D)
        for block in self.blocks:
            tokens = block(tokens, c)

        # Main velocity
        main_tokens = self.final_layer_main(tokens, c)                  # (N, T, p^2*C)
        v_main = self.unpatchify(main_tokens)                           # (N, C, H, W)

        # Proposals
        prop_imgs: List[torch.Tensor] = []
        for head in self.proposal_heads:
            ptoks = head(tokens, c)                                     # (N, T, p^2*C)
            v_k = self.unpatchify(ptoks)                                # (N, C, H, W)
            prop_imgs.append(v_k)
        v_props = torch.stack(prop_imgs, dim=1)                         # (N, K, C, H, W)

        # Refine (weights + blended velocity)
        v_ref, weights = self.refiner(v_props, v_main, c, temperature=temperature, top1=top1)

        out = {
            'v_main': v_main,
            'v_props': v_props,
            'v_ref': v_ref,
            'weights': weights,
        }
        if return_tokens:
            out['tokens'] = tokens
        return out


# ---------------------------
# Example usage in training
# ---------------------------
# model = DiT_TRM(in_channels=4, patch_size=2, hidden_size=1152, depth=28, num_heads=16,
#                 mlp_ratio=4.0, num_classes=1000, ignore_k=True, image_size=256, K=3)
# out = model(x, t, k, y, train=True, temperature=5.0, top1=False)
# v_main, v_ref, v_props, w = out['v_main'], out['v_ref'], out['v_props'], out['weights']
#
# # Example losses (you will plug in your targets/teachers):
# L_flow_main = ((v_main - v_target)**2).mean()
# L_quality   = ((v_ref  - v_target)**2).mean()
# # Optional: soft alignment between proposals and refined velocity
# with torch.no_grad():
#     # weights can be used to softly align closest proposals
#     pass



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
