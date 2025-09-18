# utils_torch/checkpoint.py
import os
import torch

def save_checkpoint(path, model, optimizer=None, step=None, ema=None, extra: dict=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"model": model.state_dict()}
    if optimizer: state["optimizer"] = optimizer.state_dict()
    if ema: state["ema"] = ema.shadow
    if step is not None: state["step"] = int(step)
    if extra: state["extra"] = extra
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, ema=None, map_location=None):
    state = torch.load(path, map_location=map_location)  # <--- add map_location
    model.load_state_dict(state["model"])
    if optimizer and "optim" in state:
        optimizer.load_state_dict(state["optim"])
    extra = state.get("extra", {})
    return state.get("step", 0), extra

