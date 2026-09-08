from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_SECRETISH = re.compile(r"(token|secret|password|api[_-]?key|credential)", re.I)
_PATHISH = re.compile(r"(path|dir|directory|file)", re.I)


def architecture_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Return a safe, human-readable architecture description for artifacts."""
    snapshot = {
        "preprocess": _component(config.get("preprocess")),
        "ocr": _component(config.get("ocr")),
        "selector": _component(config.get("selector")),
        "normalizer": _component(config.get("normalizer")),
        "runtime": _safe_value(config.get("runtime", {})),
        "weights": _safe_value(config.get("weights", {})),
    }
    return snapshot


def write_architecture_artifacts(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    snapshot = architecture_snapshot(config)
    (run_dir / "architecture.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# Experiment architecture", ""]
    for name in ("preprocess", "ocr", "selector", "normalizer"):
        component = snapshot[name]
        lines.append(f"## {name}")
        lines.append(f"- plugin: `{component.get('plugin', 'none')}`")
        params = component.get("params", {})
        if params:
            lines.append("- parameters:")
            for key, value in params.items():
                lines.append(f"  - `{key}`: `{value}`")
        else:
            lines.append("- parameters: none")
        lines.append("")
    lines.append("## runtime")
    for key, value in snapshot["runtime"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    if snapshot["weights"]:
        lines.append("## weights")
        lines.append("```json")
        lines.append(json.dumps(snapshot["weights"], ensure_ascii=False, indent=2))
        lines.append("```")
    (run_dir / "architecture.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return snapshot


def _component(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"plugin": value, "params": {}}
    if not isinstance(value, dict):
        return {"plugin": "none", "params": {}}
    return {
        "plugin": value.get("plugin", "none"),
        "params": _safe_value(value.get("params", {})),
    }


def _safe_value(value: Any, key: str = "") -> Any:
    if _SECRETISH.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _safe_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_value(item, key) for item in value]
    if _PATHISH.search(key) and isinstance(value, str):
        return Path(value).name if value else value
    return value
