
#################################################################################
#                                   DiT Configs                                  #
#################################################################################
import re

_SIZE_SPECS = {
    "XL": {"depth": 28, "hidden_size": 1152, "num_heads": 16},
    "L":  {"depth": 24, "hidden_size": 1024, "num_heads": 16},
    "B":  {"depth": 12, "hidden_size": 768,  "num_heads": 12},
    "S":  {"depth": 12, "hidden_size": 384,  "num_heads": 6},
}


def get_dit_params(model_id: str):
    """
    model_id: 'DiT-XL/2', 'DiT-L/4', 'DiT-B/8', 'DiT-S/2', ...
    returns: dict(depth=..., hidden_size=..., patch_size=..., num_heads=...)
    """
    m = re.match(r"^DiT-([A-Za-z]+)/(\d+)$", model_id.strip(), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Bad model_id '{model_id}'. Expected like 'DiT-B/4'.")
    size = m.group(1).upper()
    patch = int(m.group(2))
    if size not in _SIZE_SPECS or patch not in (2, 4, 8):
        raise ValueError(f"Unsupported size '{size}' or patch '{patch}'.")
    spec = _SIZE_SPECS[size]
    return {
        "depth": spec["depth"],
        "hidden_size": spec["hidden_size"],
        "patch_size": patch,
        "num_heads": spec["num_heads"],
    }