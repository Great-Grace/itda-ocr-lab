#!/usr/bin/env python3
"""Create a complete, auditable OCR benchmark inventory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

FIELDS = [
    "experiment", "architecture", "run_kind", "rank_eligible", "run",
    "images", "test_set", "final_em", "candidate_recall", "selection_acc",
    "sec_per_image", "images_per_sec", "peak_ram_mb", "model_weight_mb",
    "detection_ms", "recognition_ms", "ocr_ms",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_name(params: dict[str, Any], kind: str) -> str:
    name = params.get(f"text_{kind}_model_name")
    if name:
        return str(name)
    path = params.get(f"text_{kind}_model_dir")
    return Path(str(path)).name if path else "configured model"


def architecture_description(config: dict[str, Any], *, tokens_cache_used: bool) -> str:
    """Make the saved configuration useful as a human-readable experiment name."""
    ocr = config.get("ocr") or {}
    ocr_params = ocr.get("params") or {}
    preprocess = config.get("preprocess") or {}
    selector_params = (config.get("selector") or {}).get("params") or {}
    if tokens_cache_used:
        ocr_text = "OCR token cache reuse (recognition not re-run)"
    elif ocr.get("plugin") == "paddle_mobile_split":
        ocr_text = (
            f"Paddle split OCR: det={_model_name(ocr_params, 'detection')}; "
            f"rec={_model_name(ocr_params, 'recognition')}"
        )
        if ocr_params.get("recognition_input_height"):
            ocr_text += f"; rec_height={ocr_params['recognition_input_height']}"
        elif isinstance(ocr_params.get("input_shape"), list) and len(ocr_params["input_shape"]) >= 2:
            ocr_text += f"; rec_input_shape={ocr_params['input_shape']}"
    elif ocr.get("plugin") == "mock":
        ocr_text = "mock OCR fixture (not a real OCR benchmark)"
    else:
        ocr_text = f"OCR={ocr.get('plugin', 'unknown')}"
    pre_text = {
        "none": "raw image",
        "grayscale_contrast": "grayscale + contrast preprocessing",
    }.get(preprocess.get("plugin"), f"preprocess={preprocess.get('plugin', 'unknown')}")
    features: list[str] = []
    if "allow_day_first" in selector_params:
        features.append(f"day-first={'on' if selector_params['allow_day_first'] else 'off'}")
    if "allow_month_names" in selector_params:
        features.append(f"month-names={'on' if selector_params['allow_month_names'] else 'off'}")
    if selector_params.get("sanitization_mode") not in (None, "none"):
        features.append(f"sanitizer={selector_params['sanitization_mode']}")
    for key in ("keyword_weight", "position_weight"):
        if key in selector_params:
            features.append(f"{key.removesuffix('_weight')}={selector_params[key]}")
    selector = "regex date selector" + (f" ({', '.join(features)})" if features else "")
    return " | ".join((ocr_text, pre_text, selector, "date_ko_v1 normalizer"))


def classify_run(run_dir: Path, metrics: dict[str, Any], manifest: dict[str, Any]) -> tuple[str, bool]:
    """Return run type and whether it is eligible for architecture ranking."""
    input_dir = str(manifest.get("input_dir") or "")
    image_count = int(metrics.get("image_count") or 0)
    if run_dir.name.startswith("part_"):
        return "shard", False
    if input_dir.endswith("data/sample") or image_count <= 3:
        return "smoke", False
    if manifest.get("tokens_cache_used"):
        return "cached-ablation", False
    config = manifest.get("config") or {}
    if (config.get("ocr") or {}).get("plugin") == "mock":
        return "mock", False
    return "fresh-ocr", True


def _test_set(manifest: dict[str, Any]) -> str:
    if manifest.get("image_ids_path"):
        return Path(str(manifest["image_ids_path"])).stem
    return Path(str(manifest.get("input_dir") or "unknown")).name


def collect(root: Path, include_shards: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        manifest_path = run_dir / "run_manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        kind, eligible = classify_run(run_dir, metrics, manifest)
        if kind == "shard" and not include_shards:
            continue
        config = manifest.get("config") or {}
        rows.append({
            "experiment": config.get("name") or run_dir.name,
            "architecture": architecture_description(config, tokens_cache_used=bool(manifest.get("tokens_cache_used"))),
            "run_kind": kind, "rank_eligible": eligible, "run": str(run_dir),
            "images": metrics.get("image_count"), "test_set": _test_set(manifest),
            "final_em": metrics.get("final_date_exact_match"),
            "candidate_recall": metrics.get("candidate_recall"),
            "selection_acc": metrics.get("candidate_selection_accuracy"),
            "sec_per_image": None if kind == "cached-ablation" else metrics.get("sec_per_image"),
            "images_per_sec": None if kind == "cached-ablation" else metrics.get("images_per_sec"),
            "peak_ram_mb": None if kind == "cached-ablation" else metrics.get("peak_ram_mb"),
            "model_weight_mb": metrics.get("model_weight_mb"),
            "detection_ms": metrics.get("detection_ms_mean"),
            "recognition_ms": metrics.get("recognition_ms_mean"),
            "ocr_ms": metrics.get("ocr_ms_mean"),
        })
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def write_inventory(rows: list[dict[str, Any]], prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (str(row["run_kind"]), str(row["test_set"]), str(row["run"])))
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    table = [
        "# Complete OCR experiment inventory", "",
        "Every discovered metrics.json is listed. rank_eligible=yes means a fresh, non-smoke OCR run; cached parser ablations and smoke/shard measurements remain visible but are not architecture rankings.", "",
        "| Experiment | Architecture / distinguishing features | Kind | Rank? | Set | N | Final EM | Candidate recall | sec/image | Run |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append("| " + " | ".join([
            str(row["experiment"]), str(row["architecture"]), str(row["run_kind"]),
            _fmt(row["rank_eligible"]), str(row["test_set"]), _fmt(row["images"], 0),
            _fmt(row["final_em"]), _fmt(row["candidate_recall"]), _fmt(row["sec_per_image"]),
            "'" + str(row["run"]) + "'",
        ]) + " |")
    prefix.with_suffix(".md").write_text("\n".join(table) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a complete, comparable OCR benchmark inventory.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True, help="Output prefix, without .csv/.md")
    parser.add_argument("--exclude-shards", action="store_true")
    args = parser.parse_args()
    rows = collect(Path(args.runs_root), include_shards=not args.exclude_shards)
    prefix = Path(args.output)
    write_inventory(rows, prefix)
    print(json.dumps({"runs": len(rows), "csv": str(prefix.with_suffix(".csv")), "markdown": str(prefix.with_suffix(".md"))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
