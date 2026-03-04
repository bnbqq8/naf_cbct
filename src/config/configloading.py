import shutil
from pathlib import Path

import yaml


def load_config(path, default_path=None):
    """Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    """
    # Load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def save_cfg(source_cfg_path, target_dir="./tmp/"):
    """Saves config file.

    Args:
        cfg (dict): config dictionary to be saved
        path (str): path to save config file
    """
    source_cfg_path = Path(source_cfg_path)
    target_dir = Path(target_dir) / source_cfg_path.name
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_cfg_path, target_dir)
