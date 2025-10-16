import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeWarpPL(nn.Module):
    """
    Monotone piecewise-linear warp g_phi: [0,1] -> [0,1]
    - Parameterized by 'M' positive bin heights that sum to 1 (via softplus+normalization).
    - x-grid is uniform in u-space; y-grid (knots) is learned cumulative sum -> monotone by construction.
    - Returns t = g(u) and g'(u) for importance weighting.
    """
    def __init__(self, M: int = 16, init_identity: bool = True):
        super().__init__()
        # Unconstrained params -> positive via softplus
        self.beta = nn.Parameter(torch.zeros(M))
        if init_identity:
            # Start near uniform bins => g(u) ≈ u
            with torch.no_grad():
                self.beta.add_(0.0)

    @property
    def M(self):
        return self.beta.numel()

    def _heights(self):
        # Positive, normalized bin heights (sum to 1)
        alpha = F.softplus(self.beta) + 1e-8
        return alpha / alpha.sum()

    def forward(self, u: torch.Tensor):
        """
        u: (...,) in [0,1]
        returns:
          t: same shape as u, in [0,1]
          gprime: same shape as u, >= 0  (piecewise-constant slope)
        """
        device = u.device
        M = self.M

        # x-knots: uniform grid in [0,1]
        x = torch.linspace(0.0, 1.0, M + 1, device=device)  # shape (M+1,)

        # y-knots: cumulative of learned heights (strictly increasing; y[0]=0, y[-1]=1)
        heights = self._heights()                            # (M,)
        y = torch.cat([torch.zeros(1, device=device), torch.cumsum(heights, dim=0)], dim=0)

        # Bin index for each u (clamped to valid range)
        # picks the bin index i such that u ∈ [xi, xi+1]
        # Each bin has width 1/M in x-space.
        idx = torch.clamp((u * M).long(), min=0, max=M-1)    # same shape as u

        # Gather knot endpoints per-sample
        x0, x1 = x[idx], x[idx + 1]                          # (...,)
        y0, y1 = y[idx], y[idx + 1]                          # (...,)

        # Local linear interpolation weight
        w = (u - x0) / (x1 - x0 + 1e-12)                     # in [0,1]

        # Evaluate PL warp and its derivative (slope in this bin)
        t = y0 + w * (y1 - y0)                               # (...,)
        gprime = (y1 - y0) / (x1 - x0 + 1e-12)               # = M * heights[idx]

        # Numerical clamps (optional safety)
        t = t.clamp(1e-6, 1 - 1e-6)
        gprime = gprime.clamp(min=1e-8)

        return t, gprime

    def inference_times(self, K: int):
        """
        Build a K-step time schedule using the learned warp.
        - Use a uniform grid in u-space, then map via g(u) -> t_i.
        """
        device = self.beta.device
        # endpoints or midpoints both fine; midpoints can be slightly smoother
        u = torch.linspace(0.0, 1.0, K+1, device=device)
        t, _ = self.forward(u)
        return t
