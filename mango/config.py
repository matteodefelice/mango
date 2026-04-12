"""Configuration management — reads settings from a YAML file.

Lookup order:
  1. Path given explicitly via ``load(path)``
  2. ``MANGO_CONFIG`` environment variable
  3. ``./mango.yaml`` (working directory)
  4. ``~/.config/mango/config.yaml``

Any value can still be overridden programmatically after loading.
"""

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULTS: dict[str, Any] = {
    "cache_dir": str(Path.home() / ".climate_cache"),
    "edh_token": "",
}

_config: dict[str, Any] = {}


def _find_config_file() -> Path | None:
    """Return the first config file found in the lookup chain, or None."""
    env = os.environ.get("MANGO_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p

    candidates = [
        Path.cwd() / "mango.yaml",
        Path.home() / ".config" / "mango" / "config.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def load(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from *path* (or auto-discover) and return it.

    Calling ``load()`` multiple times is safe — it simply overwrites the
    in-memory config with the new file contents.
    """
    global _config

    if path is not None:
        cfg_path = Path(path)
    else:
        cfg_path = _find_config_file()

    _config = dict(_DEFAULTS)
    if cfg_path is not None and cfg_path.is_file():
        with open(cfg_path, "r") as fh:
            user = yaml.safe_load(fh) or {}
        _config.update(user)

    # Normalise types
    _config["cache_dir"] = str(_config["cache_dir"])
    return _config


def get(key: str, default: Any = None) -> Any:
    """Return a config value.  Triggers auto-load on first access."""
    if not _config:
        load()
    return _config.get(key, default)


def override(key: str, value: Any) -> None:
    """Override a config value at runtime (does NOT write to disk)."""
    if not _config:
        load()
    _config[key] = value


def cache_dir() -> Path:
    """Shortcut — return the cache directory as a Path."""
    return Path(get("cache_dir"))


def resolve_cache_dir(cache_dir: Path | None = None) -> Path:
    """Return *cache_dir* if given, otherwise fall back to config.

    Also ensures the directory exists.
    """
    resolved = Path(cache_dir) if cache_dir is not None else Path(get("cache_dir"))
    resolved.mkdir(exist_ok=True)
    return resolved


def edh_token() -> str:
    """Shortcut — return the EDH token."""
    return get("edh_token")
