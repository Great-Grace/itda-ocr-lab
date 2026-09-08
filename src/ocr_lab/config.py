from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    result = deepcopy(config)
    cursor = result
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    return result


def component_spec(config: dict[str, Any], name: str, default: str) -> tuple[str, dict[str, Any]]:
    raw = config.get(name, {}) or {}
    if isinstance(raw, str):
        return raw, {}
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a string or mapping")
    return str(raw.get("plugin", default)), dict(raw.get("params", {}) or {})
