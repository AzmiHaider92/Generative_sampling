import argparse
from dataclasses import dataclass
from typing import Optional
import yaml
from dataclasses import replace

@dataclass
class RuntimeCfg:
    dataset_name: str = "tiny-imagenet-256"
    num_classes: int = 1  #
    load_dir: Optional[str] = None
    save_dir: Optional[str] = None
    fid_stats: Optional[str] = None
    seed: int = 10
    log_interval: int = 1000
    eval_interval: int = 20000
    save_interval: int = 100000
    batch_size: int = 32
    max_steps: int = 1_000_000
    debug_overfit: int = 0
    mode: str = "train"  # train | inference
    inference_timesteps: int = 128
    inference_generations: int = 4096
    inference_cfg_scale: int = 1.0


@dataclass
class ModelCfg:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.1
    use_cosine: int = 0
    warmup: int = 0
    dropout: float = 0.0
    hidden_size: int = 64
    patch_size: int = 8
    depth: int = 2
    num_heads: int = 2
    mlp_ratio: int = 1
    class_dropout_prob: float = 0.1
    denoise_timesteps: int = 128
    cfg_scale: float = 4.0
    target_update_rate: float = 0.999
    use_ema: int = 0
    use_stable_vae: int = 1
    sharding: str = "dp"  # kept for parity; we use DDP
    t_sampling: str = "discrete-dt"
    dt_sampling: str = "uniform"
    bootstrap_cfg: int = 0
    bootstrap_every: int = 8
    bootstrap_ema: int = 1
    bootstrap_dt_bias: int = 0
    train_type: str = "shortcut"   # naive | shortcut | progressive | consistency[-distillation] | livereflow

@dataclass
class WandbCfg:
    project: str = "Generative_sampling"
    name: str = "{model_name}_{dataset_name}"
    run_id: str = "None"
    mode: str = "online"  # or "disabled"

@dataclass
class CFG:
    runtime_cfg: RuntimeCfg
    model_cfg: ModelCfg
    wandb_cfg: WandbCfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None, help='Path to YAML config')
    return p.parse_args()


def load_configs_from_file(path, runtime_cfg, model_cfg, wandb_cfg):
    """
    Whatever is in the config file overrides the dataclass defaults.
    Everything not mentioned stays as-is.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    def apply(dc, section_name):
        section = cfg.get(section_name, {})
        if not isinstance(section, dict):
            return dc
        # keep only keys that exist on the dataclass
        allowed = dc.__dataclass_fields__.keys()
        updates = {k: v for k, v in section.items() if k in allowed}
        return replace(dc, **updates)

    runtime_cfg = apply(runtime_cfg, "runtime")
    model_cfg   = apply(model_cfg, "model")
    wandb_cfg   = apply(wandb_cfg, "wandb")

    return runtime_cfg, model_cfg, wandb_cfg
