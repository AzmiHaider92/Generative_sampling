import torch
import math

EPS = 1e-5  # endpoint epsilon for the linear path


@torch.no_grad()
def get_targets(cfg, gen, ctx_tgt_dict, labels, call_model=None, step=None, force_t: float = -1, force_dt: float = -1):
    # Returns
    # ------------------------------------------------------------------------------------------------------------------
    # x_t: torch.Tensor, shapelike x0 / x1 Interpolated point between x0 and x1 at time t:
    #      x_t = (1 - (1 - eps) * t) * x0 + t * x1 ---- with eps = 1e-5 to avoid degeneracy at t=1.
    #
    # v_t : torch.Tensor, same shape as x_t
    #       The (approximate) “velocity” target of the straight-line flow:
    #       v_t = x1 - (1 - eps) * x0
    #       Note: constant in t under this parameterization; pairs with x_t to train flow-matching / shortcut operators.
    #
    # t_vec : torch.FloatTensor, shape [B]
    #         Per-sample normalized times in [0,1]. If `force_t>=0`, then all entries are that value.
    #
    # dt_base : torch.FloatTensor, shape [B]
    #           A coarse log2 step scale: dt_base = log2(denoise_timesteps)
    #           Broadcast as a per-sample scalar. This can be used as a baseline step-size feature
    #           or conditioning for schedulers / shortcut models.
    #
    # labels_eff : torch.LongTensor, shape [B]
    #              Effective labels after classifier-free dropout:
    #              with probability p=class_dropout_prob, label = num_classes (the null label);
    #              else label = original class id.

    B = ctx_tgt_dict['tgt_y'].shape[0]
    device = ctx_tgt_dict['tgt_y'].device

    T = int(cfg.model_cfg.denoise_timesteps)
    K = int(math.log2(T))  # scalar max level
    k = torch.full((B,), float(K), device=device, dtype=torch.float32)  # per-sample level code (sentinel)

    # t in [0,1]
    # Beta(2,2) via Kumaraswamy, with your RNG 'gen'
    #rho = 2.0
    #u = torch.rand(B, device=device, generator=gen)
    #t = (1 - (1 - u).pow(1 / rho)).pow(1 / rho)
    #t = t.clamp(0.02, 0.98)

    # Bins t in {0,1/128,2/128, ..., 127/128}
    t = torch.randint(low=0, high=T, size=(B,), device=device, generator=gen)
    t = t / float(T)

    t_full = t.view(B, 1, 1)

    # flow pairs
    x1 = ctx_tgt_dict['tgt_y']
    x0 = torch.randn(ctx_tgt_dict['tgt_y'].shape, dtype=ctx_tgt_dict['tgt_y'].dtype, device=device, generator=gen)

    x_t = (1.0 - (1.0 - EPS) * t_full) * x0 + t_full * x1
    v_t = x1 - x0

    return x_t, v_t, t, k
