"""
Unified config reader — single source of truth for all YAML config access.

Usage:
    from config.read_config import get_config, load_config

    # Get full config dict
    cfg = load_config()

    # Get nested value by dot-path
    port = get_config("app.fastapi.port", 1699)
    redis_url = get_config("redis.url")

Config file resolution order:
    1. CONFIG_PATH env var (for Docker / dev switching)
    2. PROJECT_CONFIG env var (legacy support)
    3. config.yaml next to this module (backend/config/config.yaml)
    4. config.yaml at project root
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    cur = cur if cur.is_dir() else cur.parent
    markers = ("pyproject.toml", ".git", "setup.cfg", "setup.py")
    for p in (cur, *cur.parents):
        if any((p / m).exists() for m in markers):
            return p
    raise RuntimeError(f"Cannot find project root from: {cur}")


def _default_config_path() -> Path:
    """Prefer config.yaml alongside this module; fall back to project root."""
    local = Path(__file__).resolve().parent / "config.yaml"
    if local.exists():
        return local
    return find_project_root() / "config.yaml"


def _resolve_config_path() -> Path:
    """Resolved config file path: CONFIG_PATH > PROJECT_CONFIG > default."""
    env_path = os.getenv("CONFIG_PATH") or os.getenv("PROJECT_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return _default_config_path()


def _resolve_overlay_path() -> Path | None:
    """Resolved optional overlay path from CONFIG_OVERLAY_PATH env."""
    env_path = os.getenv("CONFIG_OVERLAY_PATH")
    if not env_path:
        return None
    return Path(env_path).expanduser().resolve()


def _deep_merge_dict(base: Any, overlay: Any) -> Any:
    """Deep merge two YAML-loaded objects, preferring overlay values."""
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = dict(base)
        for key, value in overlay.items():
            merged[key] = _deep_merge_dict(merged.get(key), value)
        return merged
    return overlay


@lru_cache(maxsize=1)
def _load_cached() -> dict:
    """Load config from resolved path (cached singleton)."""
    path = _resolve_config_path()
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    overlay_path = _resolve_overlay_path()
    if overlay_path:
        if not overlay_path.exists():
            raise FileNotFoundError(f"overlay config file not found: {overlay_path}")
        with overlay_path.open("r", encoding="utf-8") as f:
            overlay_cfg = yaml.safe_load(f) or {}
        return _deep_merge_dict(base_cfg, overlay_cfg)

    return base_cfg


def load_config(config_path: str | Path | None = None) -> dict:
    """
    Load full YAML config as dict.

    - config_path=None  → use CONFIG_PATH/PROJECT_CONFIG env or default (cached).
    - config_path=<path> → load that specific file (not cached; for CLI --config).
    """
    if config_path is None:
        return _load_cached()
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config(path: str = "", default: Any = None) -> Any:
    """
    Get a config value by dot-separated path.

    Examples:
        get_config()                          → full config dict
        get_config("redis.url")               → "redis://localhost:6379/0"
        get_config("app.fastapi.port", 1699)  → 1699 if missing
    """
    cfg = load_config()
    if not path:
        return cfg
    cur: Any = cfg
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif isinstance(cur, list) and key.isdigit():
            idx = int(key)
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return default
        else:
            return default
    return cur if cur is not None else default


def to_json() -> str:
    """Full config as formatted JSON string (for debugging)."""
    return json.dumps(load_config(), ensure_ascii=False, indent=2)
