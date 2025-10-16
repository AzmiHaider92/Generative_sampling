import os, torch
from torch.nn.parallel import DistributedDataParallel as DDP


# --- helpers ---
def _live(m):  # unwrap DDP if needed
    return m.module if isinstance(m, DDP) else m


def _safe_load_opt(opt, state, name="opt"):
    if opt is None or state is None:
        return False
    try:
        opt.load_state_dict(state)
        return True
    except Exception as e:
        print(f"[ckpt] skip loading {name}: {e}")
        return False


# ========== SAVE ==========
def save_checkpoint(path, model, t_model=None, optimizer=None, t_optimizer=None,
              ema=None, t_ema=None, step=None, extra: dict=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    m  = _live(model)
    tm = _live(t_model) if t_model is not None else None

    state = {"schema": "compat-v1"}
    state["model"]   = m.state_dict()
    if tm is not None:
        state["t_model"] = tm.state_dict()

    if ema is not None:
        state["ema"] = ema.state_dict() if hasattr(ema, "state_dict") else ema
    if t_ema is not None:
        state["t_ema"] = t_ema.state_dict() if hasattr(t_ema, "state_dict") else t_ema

    if optimizer is not None:
        state["opt"] = optimizer.state_dict()
    if t_optimizer is not None:
        state["t_opt"] = t_optimizer.state_dict()

    if step is not None: state["step"] = int(step)
    if extra:            state["extra"] = extra

    torch.save(state, path)


# ========== LOAD (backward-compatible) ==========
def load_checkpoint(path, model, t_model=None, optimizer=None, t_optimizer=None,
              ema=None, t_ema=None, map_location="cpu",
              strict_models=True, load_optimizers=True):
    ckpt = torch.load(path, map_location=map_location)

    m  = _live(model)
    tm = _live(t_model) if t_model is not None else None

    # --- models ---
    missing, unexpected = m.load_state_dict(ckpt["model"], strict=strict_models)
    if (missing or unexpected) and strict_models:
        print(f"[ckpt] model missing={missing} unexpected={unexpected}")

    if tm is not None and "t_model" in ckpt:
        missing, unexpected = tm.load_state_dict(ckpt["t_model"], strict=strict_models)
        if (missing or unexpected) and strict_models:
            print(f"[ckpt] t_model missing={missing} unexpected={unexpected}")
    # if tm is provided but not in ckpt -> fine, you’re adding it now

    # --- EMA ---
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"]) if hasattr(ema, "load_state_dict") else None
    if t_ema is not None and "t_ema" in ckpt:
        t_ema.load_state_dict(ckpt["t_ema"]) if hasattr(t_ema, "load_state_dict") else None

    # --- optimizers (load what exists; skip otherwise) ---
    if load_optimizers:
        _safe_load_opt(optimizer,   ckpt.get("opt"),   name="optimizer")
        _safe_load_opt(t_optimizer, ckpt.get("t_opt"), name="t_optimizer")

    return ckpt.get("step", 0)
