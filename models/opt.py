# ---- 1) helpers: per-group cosine schedule ----
import math, torch
from torch import nn


# --- helpers ---
def get_group_lr(opt, name):
    for g in opt.param_groups:
        if g.get("group_name") == name:
            return g["lr"]
    return None


def mean_dit_lr(opt):
    vals = [g["lr"] for g in opt.param_groups if str(g.get("group_name","")).startswith("dit_")]
    return sum(vals)/len(vals) if vals else None


def cosine_warmup_lr(step, *, base_lr, warmup, min_lr, max_steps):
    if step < warmup:
        return base_lr * (step / max(1, warmup))
    t = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def delayed_cosine_lr(step, *, start_step, base_lr, warmup, min_lr, max_steps):
    if step < start_step:
        return 0.0
    return cosine_warmup_lr(step - start_step,
                            base_lr=base_lr, warmup=warmup,
                            min_lr=min_lr, max_steps=max_steps)


# ====== BUILD TWO OPTIMIZERS (OR NONE) ======
def build_optimizers_two(
    live_dit, live_warp_or_none,
    *,
    # DiT
    train_dit: bool,
    dit_lr: float, dit_wd: float,
    dit_warmup: int, dit_minlr_ratio: float, max_steps: int,
    # Warper
    train_warp: bool,
    warp_lr: float, warp_warmup: int, warp_minlr_ratio: float,
    warp_start_step: int = 0,      # e.g., start later (0 = start immediately)
    betas=(0.9, 0.95), eps=1e-8,
):
    opt_dit = None
    opt_warp = None

    if train_dit:
        opt_dit = torch.optim.AdamW(
            live_dit.parameters(), lr=dit_lr, weight_decay=dit_wd, betas=betas, eps=eps
        )
        # attach a tiny scheduler config for clarity
        opt_dit._sched = dict(
            kind="cosine",
            base_lr=dit_lr, warmup=dit_warmup,
            min_lr=dit_lr * dit_minlr_ratio, max_steps=max_steps
        )

    if train_warp and (live_warp_or_none is not None):
        opt_warp = torch.optim.AdamW(
            live_warp_or_none.parameters(), lr=warp_lr, weight_decay=0.0, betas=betas, eps=eps
        )
        opt_warp._sched = dict(
            kind="delayed_cosine",
            base_lr=warp_lr, warmup=warp_warmup,
            min_lr=warp_lr * warp_minlr_ratio, max_steps=max_steps,
            start_step=warp_start_step
        )

    return opt_dit, opt_warp


# ====== STEP SCHEDULERS EACH ITER ======
def step_two_schedulers(step, opt_dit, opt_warp):
    if opt_dit is not None:
        s = opt_dit._sched
        lr = cosine_warmup_lr(step, base_lr=s["base_lr"], warmup=s["warmup"],
                              min_lr=s["min_lr"], max_steps=s["max_steps"])
        opt_dit.param_groups[0]["lr"] = lr

    if opt_warp is not None:
        s = opt_warp._sched
        if s["kind"] == "delayed_cosine":
            lr = delayed_cosine_lr(step, start_step=s["start_step"],
                                   base_lr=s["base_lr"], warmup=s["warmup"],
                                   min_lr=s["min_lr"], max_steps=s["max_steps"])
        else:
            lr = cosine_warmup_lr(step, base_lr=s["base_lr"], warmup=s["warmup"],
                                  min_lr=s["min_lr"], max_steps=s["max_steps"])
        opt_warp.param_groups[0]["lr"] = lr


def optimizer(cfg, live_dit, live_warp):
    base_lr = cfg.model_cfg.lr
    warmup = cfg.model_cfg.warmup
    max_steps = cfg.runtime_cfg.max_steps
    tlr = cfg.model_cfg.t_lr
    twarp_start_time = cfg.model_cfg.t_start_warp

    # CASE 1: No warper
    if live_warp is None:
        opt_dit, opt_warp = build_optimizers_two(
            live_dit, None,
            train_dit=True, dit_lr=base_lr, dit_wd=0.05, dit_warmup=warmup, dit_minlr_ratio=0.1, max_steps=max_steps,
            train_warp=False, warp_lr=0.0, warp_warmup=0, warp_minlr_ratio=0.0
        )
    # CASE 2: Train DiT + warper together (same schedule *shape*, different LRs)
    elif not cfg.model_cfg.freeze_dit:
        opt_dit, opt_warp = build_optimizers_two(
            live_dit, live_warp,
            train_dit=True, dit_lr=base_lr, dit_wd=0.05, dit_warmup=warmup, dit_minlr_ratio=0.1, max_steps=max_steps,
            train_warp=True, warp_lr=tlr, warp_warmup=max(0, warmup // 4), warp_minlr_ratio=0.2, warp_start_step=0
    )
    # CASE 3: Freeze DiT, train warper only (optionally start later at step twarp_start_time)
    else:
        for p in live_dit.parameters(): p.requires_grad_(False)
        opt_dit, opt_warp = build_optimizers_two(
            live_dit, live_warp,
            train_dit=False, dit_lr=0.0, dit_wd=0.0, dit_warmup=0, dit_minlr_ratio=0.0, max_steps=max_steps,
            train_warp=True, warp_lr=tlr, warp_warmup=max(0, warmup // 4), warp_minlr_ratio=0.2, warp_start_step=twarp_start_time
        )

    return opt_dit, opt_warp
