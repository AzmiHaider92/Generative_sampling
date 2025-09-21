import os, torch

def save_checkpoint(path, model, optimizer=None, step=None, ema=None, extra: dict=None):
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)

    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if step is not None:
        state["step"] = int(step)
    if extra:
        state["extra"] = extra

    # ---- EMA (handle multiple implementations) ----
    if ema is not None:
        if hasattr(ema, "state_dict"):               # e.g., torch_ema.ExponentialMovingAverage or nn.Module
            state["ema"] = ema.state_dict()
            state["ema_format"] = "state_dict"
        elif isinstance(ema, torch.nn.Module):       # separate EMA model
            state["ema"] = ema.state_dict()
            state["ema_format"] = "module_state_dict"
        elif hasattr(ema, "shadow"):                 # old custom wrappers storing a dict of tensors
            state["ema"] = ema.shadow
            state["ema_format"] = "shadow"
        elif hasattr(ema, "shadow_params"):          # list of tensors
            state["ema"] = [p.detach().cpu() for p in ema.shadow_params]
            state["ema_format"] = "shadow_params"
        # else: silently skip if unknown

    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, ema=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    # ---- restore EMA if provided ----
    if ema is not None and "ema" in ckpt:
        fmt = ckpt.get("ema_format", "state_dict")
        if fmt in ("state_dict", "module_state_dict") and hasattr(ema, "load_state_dict"):
            ema.load_state_dict(ckpt["ema"])
        elif hasattr(ema, "shadow"):
            ema.shadow = ckpt["ema"]
        elif hasattr(ema, "shadow_params"):
            for p, q in zip(ema.shadow_params, ckpt["ema"]):
                p.data.copy_(q)

    step = ckpt.get("step", 0)
    extra = ckpt.get("extra", None)
    return step, extra

