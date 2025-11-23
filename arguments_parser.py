import argparse
from dataclasses import dataclass
from typing import Optional
import yaml
from dataclasses import replace

@dataclass
class RuntimeCfg:
    dataset_name: str = "tiny-imagenet-256"
    dataset_root_dir: str = "./data"
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
    img_size: int = 256
    mode: str = "train"  # train | inference
    inference_timesteps: int = 30
    inference_cfg_scale: int = 1.0
    inference_num_generations: int = 8
    calc_fid: int = 0


@dataclass
class ModelCfg:
    model_id: str = "DiT-B/2"
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1

    lr: float = 1e-4
    denoise_timesteps: int = 128
    cfg_scale: float = 1.5
    warmup: int = 10_000
    use_ema: int = 1
    use_stable_vae: int = 1

    train_type: str = "shortcut"   # flow_matching | shortcut | progressive | consistency[-distillation] | livereflow

    # bootstrapping in shortcut models
    bootstrap_every: int = 8
    bootstrap_cfg: int = 0
    bootstrap_ema: int = 1
    bootstrap_dt_bias: int = 0

    # policy learning
    lr_policy: float = 1e-4

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
