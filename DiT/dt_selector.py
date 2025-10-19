import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128, max_period: int = 10_000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] in [0,1]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=device).float() / max(1, half - 1)
        )
        angles = t[:, None] * freqs[None, :] * 2 * math.pi
        emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class AttnPool1D(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) / math.sqrt(dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        q = self.q.expand(B, 1, D)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)  # [B, D]


class DtSelector(nn.Module):
    """
    Token/latent-based dt selector with label y.

    mode='discrete': returns logits [B, K] over k ∈ {0..K-1}
    mode='continuous': returns raw [B, 1] (caller maps to dt then discretizes if desired)
    """
    def __init__(
        self,
        K: int,
        token_dim: int,                 # D of tokens (or C if using [B,C,H,W])
        t_dim: int = 128,
        class_dim: int = 64,
        num_classes: int = 1000,        # set to your dataset classes
        mode: str = "discrete",
        n_heads: int = 4,
    ):
        super().__init__()
        assert mode in ("discrete", "continuous")
        self.K = K
        self.mode = mode
        self.num_classes = num_classes

        self.token_ln = nn.LayerNorm(token_dim)
        self.pool = AttnPool1D(token_dim, n_heads=n_heads)
        self.t_emb = TimeEmbedding(t_dim)
        num_classes_ = num_classes + (num_classes > 1)
        self.class_emb = nn.Embedding(num_classes_, class_dim)  # +1 for unconditional idx=num_classes

        fused_dim = token_dim + t_dim + class_dim
        hidden = max(256, fused_dim)  # heuristic; feel free to change

        if mode == "discrete":
            self.head = nn.Sequential(
                nn.Linear(fused_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, K),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(fused_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, N, D] or [B, C, H, W]; return [B, N, D]
        if x.dim() == 3:
            return x  # [B, N, D]
        if x.dim() == 4:
            B, C, H, W = x.shape
            return x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        raise ValueError(f"Unsupported x_t shape: {x.shape}")

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        x_t: [B, C, H, W] or [B, N, D]
        t:   [B]
        y:   [B] int64 (use num_classes for unconditional)
        """
        tokens = self._to_tokens(x_t)     # [B, N, D]
        tokens = self.token_ln(tokens)
        pooled = self.pool(tokens)        # [B, D]

        emb_t = self.t_emb(t)             # [B, t_dim]
        emb_y = self.class_emb(y)         # [B, class_dim]

        h = torch.cat([pooled, emb_t, emb_y], dim=-1)  # [B, fused_dim]
        return self.head(h)                # [B, K] or [B, 1]
