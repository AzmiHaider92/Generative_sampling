import torch
import torch.nn as nn


class PolicyWithDiTEmbedders(nn.Module):
    """
    Reuse DiT embedders to build a Δt policy.

    Expected DiT pieces you pass in:
      - patch_embed:  x[B,C,H,W] -> tokens[B, N, D]  (or features[B, D])
      - t_embedder:   t[B] -> t_emb[B, T]
      - y_embedder:   y[B] -> y_emb[B, Y] (optional; pass None if unused)

    We pool tokens with mean to get x_emb[B, D]. If your DiT already has a CLS
    token or a nice feature tap (e.g., last block output), you can pass a
    callable that returns [B, D] directly instead of patch_embed.
    """
    def __init__(
        self,
        x_embedder: nn.Module,         # or a callable to get x features
        t_embedder: nn.Module,          # your existing TimestepEmbedder
        pose_embed: nn.Parameter,
        y_embedder: nn.Module | None,   # your existing class embed (optional)
        head_hidden: int = 256,
        T: int = 128,
    ):
        super().__init__()
        self.x_embedder = x_embedder
        self.embedders_outdim = x_embedder.proj.out_channels
        self.pose_embed = pose_embed

        self.t_embedder = t_embedder
        self.y_embedder = y_embedder

        self.dt_min = 1./T
        self.dt_max = 1

        # Infer dims lazily on first forward if you like; here we build a generic head that
        # will be re-initialized after the first shape pass if needed. For simplicity,
        # we ask the user to call `init_head(example_shapes)` once, or we do a quick probe.
        self.head_hidden = head_hidden
        self.head = self.head = nn.Sequential(
            nn.Linear(3*self.embedders_outdim, self.head_hidden), nn.SiLU(),
            nn.Linear(self.head_hidden, self.head_hidden), nn.SiLU(),
            nn.Linear(self.head_hidden, 1),
        )

    def _cap_dt(self, raw01: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cap = torch.minimum(torch.full_like(raw01, self.dt_max), 1.0 - t)
        cap = torch.clamp(cap, min=self.dt_min)
        return self.dt_min + (cap - self.dt_min) * raw01

    def _x_embed(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Use DiT's patch_embed (or feature tap). We expect:
          - If it returns [B, N, D], we mean-pool over tokens.
          - If it returns [B, D], we keep it.
        """
        out = self.x_embedder(x_t) + self.pose_embed
        out = out.mean(dim=1)
        return out

    def _t_embed(self, t: torch.Tensor) -> torch.Tensor:
        te = self.t_embedder(t)
        return te

    def _y_embed(self, y: torch.Tensor | None) -> torch.Tensor | None:
        ye = self.y_embedder(y, train=False)
        return ye

    @torch.no_grad()
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Deterministic Δt (use at inference).
        """
        xh = self._x_embed(x_t)     # [B, Dx]
        th = self._t_embed(t)       # [B, Dt]
        yh = self._y_embed(y)       # [B, Dy] or None
        h  = torch.cat([xh, th, yh], dim=1)

        z = self.head(h).squeeze(-1)  # [B]
        raw01 = torch.sigmoid(z)
        return self._cap_dt(raw01, t)


def train_policy(dit):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dummy random batch with BCHW layout
    B, C, H, W = 8, 4, 32, 32
    T = 128
    x_t = torch.randn(B, C, H, W, device=device)
    t   = torch.rand(B, device=device) * 0.95  # times in [0, 0.95)
    y   = torch.zeros(B, device=device, dtype=torch.int64)

    # policy: input dim = feats(2) + t(1)
    policy = PolicyWithDiTEmbedders(
        x_embedder=dit.x_embedder,  # reuse
        pose_embed=dit.pos_embed,
        t_embedder=dit.t_embedder,  # reuse
        y_embedder=dit.y_embedder,  # or None if unconditional
        T=T
    ).to(device)

    # deterministic Δt (for inference)
    with torch.no_grad():
        dt = policy(x_t, t, y)  # deterministic Δt            # [B]
    print("Deterministic Δt:", dt.detach().cpu().numpy())
