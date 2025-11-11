import math
import torch

EPS = 1e-5


def logit_normal_timestep_sample(P_mean: float, P_std: float, num_samples: int, device: torch.device) -> torch.Tensor:
    rnd_normal = torch.randn((num_samples,), device=device)
    time = torch.sigmoid(rnd_normal * P_std + P_mean)
    time = torch.clip(time, min=0.0, max=1.0)
    return time


def sample_two_timesteps(num_samples: int, device: torch.device):
    t, r = sample_two_timesteps_t_r_v0(num_samples, device=device)
    return t, r


def sample_two_timesteps_t_r_v0(num_samples: int, device: torch.device):
    """
    Sampler (t, r): independently sample t and r, with post-processing.
    Version 0: used in paper.
    """
    # step 1: sample two independent timesteps
    t = logit_normal_timestep_sample(-2, 2, num_samples, device=device)
    r = logit_normal_timestep_sample(-2, 2, num_samples, device=device)

    # step 2: ensure t >= r
    t, r = torch.maximum(t, r), torch.minimum(t, r)

    # step 3: make t and r different with a probability of args.ratio
    prob = torch.rand(num_samples, device=device)
    mask = prob < 1 - 0.75
    r = torch.where(mask, t, r)

    return t, r


@torch.no_grad()
def get_targets(
    cfg,
    gen,
    images,
    labels,
    call_model_fn,           # expects: call_model_fn(x, t, k, labels, train=False)
    step,
    force_t: float = -1.0,
    force_dt: float = -1.0,
):
    """
    Shortcut target builder with **k-level conditioning**.

    - We still sample/compute dt to advance states, but the model is conditioned on
      k = -log2(dt). The teacher uses k+1 (two half-steps).
    - Flow Matching part uses a constant k corresponding to dt_min.

    Returns:
      x_t_out, v_t_out, t_out, k_out, labels_out, info
    """
    device = images.device
    B = images.shape[0]

    # ---- Config ----
    class_dropout_prob = cfg.model_cfg.class_dropout_prob
    num_classes = cfg.runtime_cfg.num_classes

    drop = torch.bernoulli(torch.full((labels.shape[0],), class_dropout_prob, device=device)).bool()
    labels_dropped = labels.clone().long()
    labels_dropped[drop] = num_classes if num_classes > 1 else 0

    t, r = sample_two_timesteps(B, device)
    t_full = t.view(B, 1, 1, 1)
    r_full = r.view(B, 1, 1, 1)

    x0_fm = torch.randn(images.shape, dtype=images.dtype, device=images.device, generator=gen)
    x_t = (1.0 - (1.0 - EPS) * t_full) * x0_fm + t_full * images
    v_t = images - (1.0 - EPS) * x0_fm

    dtdt = torch.ones_like(t)
    drdt = torch.zeros_like(r)

    # define network function
    def u_func(z, t, r):
        h = t - r
        return call_model_fn(z, t.view(-1), h.view(-1), labels_dropped)


    # u_pred, dudt = torch.func.jvp(u_func, (z, t, r), (v, dtdt, drdt))

    eps = 1e-3
    with torch.no_grad():
        u0 = u_func(x_t, t, r)
        u1 = u_func(x_t + eps * v_t,
                    t + eps * dtdt,
                    r + eps * drdt)
        dudt = (u1 - u0) / eps
        u_tgt = (v_t - (t_full - r_full) * dudt).detach()

    return x_t, u_tgt, t, t-r, labels_dropped, None
