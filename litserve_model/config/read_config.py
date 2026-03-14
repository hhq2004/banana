import os
import yaml
from pathlib import Path

_config = {}

def load_config(config_path=None):
    global _config
    if config_path is None:
        # Default to litserve_model/config/config.yaml or similar
        config_path = Path(__file__).resolve().parent / "config.yaml"
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f) or {}
    return _config

def get_config(key, default=None):
    global _config
    if not _config:
        load_config()
    
    parts = key.split(".")
    val = _config
    for p in parts:
        if isinstance(val, dict) and p in val:
            val = val[p]
        else:
            return default
    return val
