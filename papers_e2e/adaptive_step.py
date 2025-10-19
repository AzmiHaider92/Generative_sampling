import math
import torch

EPS = 1e-5  # endpoint epsilon for the linear path

"""
Mine
"""

# --- Example selector head signatures you might have ---
# Discrete:  dt_selector(x_t, t) -> logits over K=log2(T) levels
# Continuous: dt_selector(x_t, t) -> raw scalar; map via sigmoid to (min_dt, 1]


def _predict_dt_base_from_selector(dt_selector, x_t, t, K, *, mode="discrete",
                                   exploration_p=0.1, gen=None):
    """
    Returns: dt_base: LongTensor [B_b] with values in {0..K-1}
    mode:
      - "discrete": dt_selector returns [B_b, K] logits
      - "continuous": dt_selector returns [B_b, 1] raw; map to dt in (2^{-K}, 1], then to k=round(log2(1/dt))
    exploration_p: with this prob, sample from the uniform baseline (avoids collapse)
    """
    B_b = x_t.shape[0]
    device = x_t.device

    # exploration mask
    explore = torch.rand(B_b, device=device, generator=gen) < exploration_p

    if mode == "discrete":
        with torch.no_grad():
            logits = dt_selector(x_t, t)          # [B_b, K]
            # greedy pick
            k_pred = torch.argmax(logits, dim=-1).to(torch.long)  # [B_b]
            # uniform fallback for exploration
            k_rand = torch.randint(0, K, (B_b,), device=device, generator=gen)
            dt_base = torch.where(explore, k_rand, k_pred)

    elif mode == "continuous":
        with torch.no_grad():
            raw = dt_selector(x_t, t).squeeze(-1)  # [B_b]
            # map to (min_dt, 1]; min_dt=2^{-K} to align with your grid
            min_dt = 2.0 ** (-K)
            dt_cont = torch.sigmoid(raw) * (1.0 - min_dt) + min_dt  # (min_dt, 1]
            k_float = torch.log2(1.0 / dt_cont).clamp(min=0.0, max=float(K - 1))
            k_pred = torch.round(k_float).to(torch.long)
            # uniform fallback for exploration
            k_rand = torch.randint(0, K, (B_b,), device=device, generator=gen)
            dt_base = torch.where(explore, k_rand, k_pred)
    else:
        raise ValueError(f"Unknown mode={mode}")

    return dt_base


def get_targets(cfg, gen, images, labels, call_model_fn, step,
                force_t: float = -1.0, force_dt: float = -1.0,
                dt_selector=None, selector_mode: str = "discrete",
                selector_exploration_p: float = 0.1):
    """
    Adds optional dt selection via a small network `dt_selector`.
    If provided, it overrides the bootstrap `dt_base` sampling (unless `force_dt != -1`).
    """
    device = images.device
    B = images.shape[0]
    denoise_timesteps = int(cfg.model_cfg.denoise_timesteps)  # e.g. 128
    bootstrap_every = cfg.model_cfg.bootstrap_every
    cfg_scale = cfg.model_cfg.cfg_scale
    class_dropout_prob = cfg.model_cfg.class_dropout_prob
    num_classes = cfg.runtime_cfg.num_classes

    # ---------- 1) Sample or predict dt_base (bootstrap) ----------
    bootstrap_batchsize = B // bootstrap_every
    K = int(math.log2(denoise_timesteps))  # number of levels

    if force_dt != -1:
        dt_base = torch.full((bootstrap_batchsize,), int(force_dt), device=device, dtype=torch.long)
        num_dt_cfg = max(1, bootstrap_batchsize // max(1, K))
    else:
        if dt_selector is None:
            # === original baseline path ===
            if cfg.model_cfg.bootstrap_dt_bias == 0:
                levels = torch.arange(K - 1, -1, -1, device=device, dtype=torch.long)
                reps = max(1, bootstrap_batchsize // max(1, K))
                dt_base = levels.repeat_interleave(reps)
                if dt_base.numel() < bootstrap_batchsize:
                    pad = torch.zeros(bootstrap_batchsize - dt_base.numel(), device=device, dtype=torch.long)
                    dt_base = torch.cat([dt_base, pad], dim=0)
                dt_base = dt_base[:bootstrap_batchsize]
                num_dt_cfg = bootstrap_batchsize // max(1, K)
            else:
                short_levels = torch.arange(K - 3, -1, -1, device=device, dtype=torch.long)
                reps = max(1, (bootstrap_batchsize // 2) // max(1, K))
                dt_base = short_levels.repeat_interleave(reps)
                part = bootstrap_batchsize // 4
                dt_base = torch.cat([dt_base,
                                     torch.ones(part, device=device, dtype=torch.long),
                                     torch.zeros(part, device=device, dtype=torch.long)], dim=0)
                if dt_base.numel() < bootstrap_batchsize:
                    pad = torch.zeros(bootstrap_batchsize - dt_base.numel(), device=device, dtype=torch.long)
                    dt_base = torch.cat([dt_base, pad], dim=0)
                dt_base = dt_base[:bootstrap_batchsize]
                num_dt_cfg = max(1, (bootstrap_batchsize // 2) // max(1, K))
        else:
            # === selector-driven path ===
            # We need x_t to run the selector; build x_t with a provisional t first (doesn't depend on dt)
            # Sample t on a per-level grid AFTER we have dt_base, but the selector only needs (x_t, t).
            # For the first pass, use a uniform t in [0,1) so the selector can see the current state.
            t_probe = torch.rand(bootstrap_batchsize, device=device, generator=gen)
            t_probe_full = t_probe.view(bootstrap_batchsize, *([1] * (images.ndim - 1)))

            x_1_probe = images[:bootstrap_batchsize]
            x_0_probe = torch.randn(x_1_probe.shape, dtype=x_1_probe.dtype, device=x_1_probe.device, generator=gen)

            x_t_probe = (1.0 - (1.0 - EPS) * t_probe_full) * x_0_probe + t_probe_full * x_1_probe

            # Predict k (dt_base) from the probe state:
            dt_base = _predict_dt_base_from_selector(
                dt_selector, x_t_probe, t_probe, K,
                mode=selector_mode, exploration_p=selector_exploration_p, gen=gen
            )
            # Choose a reasonable duplication count per level for CFG bootstrap:
            num_dt_cfg = max(1, bootstrap_batchsize // max(1, K))

    dt = 1.0 / (2.0 ** dt_base.to(torch.float32))     # (B_b,)
    dt_base_bootstrap = dt_base + 1                   # use exact half-step level
    dt_bootstrap = dt / 2.0

    # ---------- 2) Sample t on the level-specific grid ----------
    dt_sections = 2 ** dt_base                        # [1,2,4,...]
    max_section = int(dt_sections.max().item()) if dt_sections.numel() > 0 else 1
    t_raw = torch.randint(0, max_section, (bootstrap_batchsize,), device=device, generator=gen)
    t_mod = (t_raw % torch.clamp(dt_sections, min=1))
    t = t_mod.to(torch.float32) / torch.clamp(dt_sections.to(torch.float32), min=1.0)
    if force_t != -1:
        t = torch.full_like(t, float(force_t))
    t_full = t.view(bootstrap_batchsize, *([1] * (images.ndim - 1)))

    # ---------- 3) Bootstrap targets ----------
    x_1 = images[:bootstrap_batchsize]
    x_0 = torch.randn(x_1.shape, dtype=x_1.dtype, device=x_1.device, generator=gen)

    x_t = (1.0 - (1.0 - EPS) * t_full) * x_0 + t_full * x_1
    bst_labels = labels[:bootstrap_batchsize]

    if not cfg.model_cfg.bootstrap_cfg:
        v_b1 = call_model_fn(x_t, t, dt_base_bootstrap, bst_labels, train=False)
        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap.view(-1, *([1] * (images.ndim - 1))) * v_b1
        x_t2 = torch.clamp(x_t2, -4.0, 4.0)
        v_b2 = call_model_fn(x_t2, t2, dt_base_bootstrap, bst_labels, train=False)
        v_target = 0.5 * (v_b1 + v_b2)
    else:
        x_t_extra = torch.cat([x_t, x_t[:num_dt_cfg]], dim=0)
        t_extra = torch.cat([t, t[:num_dt_cfg]], dim=0)
        dt_base_extra = torch.cat([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], dim=0)
        labels_extra = torch.cat([bst_labels, torch.full((num_dt_cfg,), num_classes, device=device, dtype=torch.long)], dim=0)

        v_b1_raw = call_model_fn(x_t_extra, t_extra, dt_base_extra, labels_extra, train=False)
        v_b1_cond = v_b1_raw[:x_1.shape[0]]
        v_b1_uncond = v_b1_raw[x_1.shape[0]:]
        v_cfg = v_b1_uncond + cfg_scale * (v_b1_cond[:num_dt_cfg] - v_b1_uncond)
        v_b1 = torch.cat([v_cfg, v_b1_cond[num_dt_cfg:]], dim=0)

        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap.view(-1, *([1] * (images.ndim - 1))) * v_b1
        x_t2 = torch.clamp(x_t2, -4.0, 4.0)

        x_t2_extra = torch.cat([x_t2, x_t2[:num_dt_cfg]], dim=0)
        t2_extra = torch.cat([t2, t2[:num_dt_cfg]], dim=0)

        v_b2_raw = call_model_fn(x_t2_extra, t2_extra, dt_base_extra, labels_extra, train=False)
        v_b2_cond = v_b2_raw[:x_1.shape[0]]
        v_b2_uncond = v_b2_raw[x_1.shape[0]:]
        v_b2_cfg = v_b2_uncond + cfg_scale * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
        v_b2 = torch.cat([v_b2_cfg, v_b2_cond[num_dt_cfg:]], dim=0)

        v_target = 0.5 * (v_b1 + v_b2)

    v_target = torch.clamp(v_target, -4.0, 4.0)
    bst_v  = v_target
    bst_dt = dt_base.clone()
    bst_t  = t.clone()
    bst_xt = x_t
    bst_l  = bst_labels

    # ---------- 4) Flow-Matching targets ----------
    drop = torch.bernoulli(torch.full((labels.shape[0],), class_dropout_prob, device=device)).bool()
    labels_dropped = labels.clone().long()
    labels_dropped[drop] = num_classes if num_classes > 1 else 0

    t_flow = torch.randint(0, denoise_timesteps, (B,), device=device, generator=gen).float() / float(denoise_timesteps)
    if force_t != -1:
        t_flow = torch.full_like(t_flow, float(force_t))
    t_flow_full = t_flow.view(B, *([1] * (images.ndim - 1)))

    x1_flow = images
    x0_flow = torch.randn(x1_flow.shape, dtype=x1_flow.dtype, device=x1_flow.device, generator=gen)

    x_t_flow = (1.0 - (1.0 - EPS) * t_flow_full) * x0_flow + t_flow_full * x1_flow
    v_t_flow = x1_flow - (1.0 - EPS) * x0_flow

    dt_flow_int = K
    dt_base_flow = torch.full((B,), dt_flow_int, device=device, dtype=torch.long)

    # ---------- 5) Merge ----------
    bst_size = B // bootstrap_every
    bst_size_data = B - bst_size

    x_t_out     = torch.cat([bst_xt,           x_t_flow[:bst_size_data]], dim=0)
    v_t_out     = torch.cat([bst_v,            v_t_flow[:bst_size_data]], dim=0)
    t_out       = torch.cat([bst_t,            t_flow[:bst_size_data]],   dim=0)
    dt_base_out = torch.cat([bst_dt,           dt_base_flow[:bst_size_data]], dim=0)
    labels_out  = torch.cat([bst_l,            labels_dropped[:bst_size_data]], dim=0)

    return x_t_out, v_t_out, t_out, dt_base_out, labels_out, None
