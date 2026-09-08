from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import component_spec
from .contracts import OCRToken
from .architecture import write_architecture_artifacts
from .modules import NORMALIZERS, OCR_BACKENDS, PREPROCESSORS, SELECTORS
from .reporting import generate_reports


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run_pipeline(
    config: dict[str, Any],
    input_dir: str | Path,
    output_dir: str | Path,
    labels_path: str | Path | None = None,
    max_images: int | None = None,
    tokens_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    architecture = write_architecture_artifacts(run_dir, config)
    image_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if max_images:
        image_paths = image_paths[:max_images]

    preprocess_name, preprocess_params = component_spec(config, "preprocess", "none")
    ocr_name, ocr_params = component_spec(config, "ocr", "mock")
    selector_name, selector_params = component_spec(config, "selector", "keyword_regex")
    normalizer_name, normalizer_params = component_spec(config, "normalizer", "date_ko_v1")

    if preprocess_name not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocessor '{preprocess_name}'. Available: {sorted(PREPROCESSORS)}")
    if ocr_name not in OCR_BACKENDS:
        raise ValueError(f"Unknown OCR backend '{ocr_name}'. Available: {sorted(OCR_BACKENDS)}")
    if selector_name not in SELECTORS or normalizer_name not in NORMALIZERS:
        raise ValueError("Unknown selector or normalizer plugin. Check configs and module catalog.")

    preprocessor = PREPROCESSORS[preprocess_name](**preprocess_params)
    # The runner's device flag must reach the backend, not merely be recorded
    # in the manifest. Backend adapters may translate this to their own API.
    ocr_params = {**ocr_params, "runtime_device": (config.get("runtime") or {}).get("device", "cpu")}
    ocr = OCR_BACKENDS[ocr_name](**ocr_params)
    selector = SELECTORS[selector_name](**selector_params)
    normalizer = NORMALIZERS[normalizer_name](**normalizer_params)

    # Load cached tokens if available
    cached_tokens: dict[str, list[OCRToken]] = {}
    if tokens_cache_path and Path(tokens_cache_path).exists():
        with Path(tokens_cache_path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cached_tokens[row["image_id"]] = [OCRToken(**t) for t in row.get("tokens", [])]

    predictions = []
    token_rows = []
    review_rows = []
    latencies = []
    preprocessed_dir = run_dir / "preprocessed"

    for image_path in image_paths:
        started = time.perf_counter()
        image_id = image_path.stem

        if image_id in cached_tokens:
            tokens: list[OCRToken] = cached_tokens[image_id]
        else:
            processed_path = preprocessor.preprocess(image_path, preprocessed_dir)
            tokens = ocr.extract(processed_path)

        candidate = selector.select(tokens)
        prediction = normalizer.normalize(image_id, candidate)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        predictions.append(asdict(prediction))
        token_rows.append({"image_id": image_id, "tokens": [asdict(token) for token in tokens]})
        review_rows.append({
            "image_id": image_id,
            "ocr_text": " | ".join(token.text for token in tokens),
            "candidate": candidate.raw_text if candidate else "NONE",
            "candidate_score": candidate.score if candidate else 0.0,
            "final_date": prediction.final_date,
            "confidence": prediction.confidence,
            "elapsed_ms": round(elapsed_ms, 3),
            "evidence": " | ".join(prediction.evidence),
        })

    _write_csv(run_dir / "predictions.csv", predictions, ["image_id", "year", "month", "day", "final_date"])
    _write_csv(run_dir / "review.csv", review_rows, list(review_rows[0]) if review_rows else ["image_id"])
    (run_dir / "ocr_tokens.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in token_rows) + ("\n" if token_rows else ""), encoding="utf-8")
    metrics = _metrics(predictions, labels_path, latencies)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Generate human-readable Markdown summary and self-contained review HTML viewer
    generate_reports(run_dir, predictions, review_rows, metrics, labels_path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "input_dir": str(input_path),
        "image_count": len(image_paths),
        "device": (config.get("runtime") or {}).get("device", "cpu"),
        "preprocess": preprocess_name,
        "ocr_backend": ocr_name,
        "selector": selector_name,
        "normalizer": normalizer_name,
        "tokens_cache_used": bool(cached_tokens),
        "architecture": architecture,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics



def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metrics(predictions: list[dict[str, Any]], labels_path: str | Path | None, latencies: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "image_count": len(predictions),
        "none_count": sum(item["final_date"] == "NONE" for item in predictions),
        "none_rate": (sum(item["final_date"] == "NONE" for item in predictions) / len(predictions)) if predictions else 0.0,
        "latency_ms_mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "latency_ms_p95": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0,
    }
    if labels_path:
        labels = {}
        with Path(labels_path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                labels[str(row["image_id"])] = str(row["final_date"])
        comparable = [item for item in predictions if item["image_id"] in labels]
        result["labeled_count"] = len(comparable)
        result["final_date_exact_match"] = (sum(item["final_date"] == labels[item["image_id"]] for item in comparable) / len(comparable)) if comparable else 0.0
    return result
