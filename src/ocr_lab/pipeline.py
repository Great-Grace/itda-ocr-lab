from __future__ import annotations

import csv
import json
import os
import re
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .architecture import write_architecture_artifacts
from .config import component_spec
from .contracts import OCRToken
from .modules import NORMALIZERS, OCR_BACKENDS, PREPROCESSORS, SELECTORS
from .modules.date_candidates import candidate_final_date, parse_final_date
from .reporting import generate_reports

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_DATEISH = re.compile(r"\d{2,4}\s*[.\-/년]\s*\d{1,2}|\d{6,8}")


def run_pipeline(config: dict[str, Any], input_dir: str | Path, output_dir: str | Path,
                 labels_path: str | Path | None = None, max_images: int | None = None,
                 tokens_cache_path: str | Path | None = None, shard_index: int = 0,
                 shard_count: int = 1, image_ids_path: str | Path | None = None) -> dict[str, Any]:
    input_path, run_dir = Path(input_dir), Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    architecture = write_architecture_artifacts(run_dir, config)
    image_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if image_ids_path:
        selected_ids = {
            line.strip().rsplit(".", 1)[0]
            for line in Path(image_ids_path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        image_paths = [path for path in image_paths if path.stem in selected_ids]
    if max_images:
        image_paths = image_paths[:max_images]
    if shard_count > 1:
        if not 0 <= shard_index < shard_count:
            raise ValueError("shard_index must be within [0, shard_count)")
        image_paths = image_paths[shard_index::shard_count]

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

    runtime = config.get("runtime") or {}
    cached_tokens = _load_cached_tokens(tokens_cache_path)
    needs_ocr = any(path.stem not in cached_tokens for path in image_paths)
    preprocessor = PREPROCESSORS[preprocess_name](**preprocess_params)
    ocr = None
    if needs_ocr:
        runtime_ocr_params = {**ocr_params, "runtime_device": runtime.get("device", "cpu")}
        if runtime.get("threads") and ocr_name == "paddle_mobile_split":
            runtime_ocr_params["cpu_threads"] = int(runtime["threads"])
        ocr = OCR_BACKENDS[ocr_name](**runtime_ocr_params)
    selector = SELECTORS[selector_name](**selector_params)
    normalizer = NORMALIZERS[normalizer_name](**normalizer_params)

    predictions, token_rows, review_rows, latencies, stage_rows = [], [], [], [], []
    preprocessed_dir = run_dir / "preprocessed"
    started_run, peak_rss = time.perf_counter(), _rss_bytes()
    for idx, image_path in enumerate(image_paths, start=1):
        started = time.perf_counter()
        image_id, stage = image_path.stem, {"preprocess_ms": 0.0, "detection_ms": None,
            "recognition_ms": None, "ocr_ms": 0.0, "candidate_generation_ms": 0.0,
            "selection_ms": 0.0, "normalization_ms": 0.0}
        if image_id in cached_tokens:
            tokens = cached_tokens[image_id]
        else:
            if ocr is None:
                raise RuntimeError(f"OCR token cache is missing image_id={image_id}")
            t = time.perf_counter()
            processed_path = preprocessor.preprocess(image_path, preprocessed_dir)
            stage["preprocess_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            tokens = ocr.extract(processed_path)
            stage["ocr_ms"] = (time.perf_counter() - t) * 1000
            backend_metrics = getattr(ocr, "last_stage_metrics", {}) or {}
            stage["detection_ms"], stage["recognition_ms"] = backend_metrics.get("detection_ms"), backend_metrics.get("recognition_ms")
        peak_rss = max(peak_rss or 0, _rss_bytes() or 0) or None
        t = time.perf_counter()
        candidates = selector.candidates(tokens) if hasattr(selector, "candidates") else []
        stage["candidate_generation_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        candidate = selector.select_candidates(candidates, tokens) if hasattr(selector, "select_candidates") else selector.select(tokens)
        stage["selection_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        prediction = normalizer.normalize(image_id, candidate)
        stage["normalization_ms"] = (time.perf_counter() - t) * 1000
        elapsed = (time.perf_counter() - started) * 1000
        stage_rows.append(stage); latencies.append(elapsed)
        predictions.append(asdict(prediction))
        token_rows.append({"image_id": image_id, "tokens": [asdict(token) for token in tokens]})
        review_rows.append({"image_id": image_id, "ocr_text": " | ".join(t.text for t in tokens),
            "token_count": len(tokens), "candidate_count": len(candidates),
            "candidate_set": " | ".join(candidate_final_date(c) or c.raw_text for c in candidates),
            "candidate": candidate.raw_text if candidate else "NONE",
            "selected_candidate_normalized": candidate_final_date(candidate) if candidate else "NONE",
            "candidate_score": candidate.score if candidate else 0.0, "final_date": prediction.final_date,
            "confidence": prediction.confidence, "elapsed_ms": round(elapsed, 3),
            **{key: round(value, 3) if value is not None else "" for key, value in stage.items()},
            "evidence": " | ".join(prediction.evidence)})
        if idx % 25 == 0 or idx == len(image_paths) or idx == 1:
            print(f"[{idx}/{len(image_paths)}] processed image {image_id} ({elapsed:.1f}ms)", flush=True)

    _write_csv(run_dir / "predictions.csv", predictions, ["image_id", "year", "month", "day", "final_date"])
    _write_csv(run_dir / "review.csv", review_rows, list(review_rows[0]) if review_rows else ["image_id"])
    (run_dir / "ocr_tokens.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in token_rows) + ("\n" if token_rows else ""), encoding="utf-8")
    metrics = _metrics(predictions, review_rows, labels_path, latencies, stage_rows, config,
                       time.perf_counter() - started_run, peak_rss)
    # Metrics enrichment adds candidate-recall and error-category columns.
    _write_csv(run_dir / "review.csv", review_rows, list(review_rows[0]) if review_rows else ["image_id"])
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_reports(run_dir, predictions, review_rows, metrics, labels_path, architecture=architecture)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config,
        "input_dir": str(input_path), "image_count": len(image_paths), "device": runtime.get("device", "cpu"),
        "runtime_threads": runtime.get("threads"), "preprocess": preprocess_name, "ocr_backend": ocr_name,
        "selector": selector_name, "normalizer": normalizer_name, "tokens_cache_used": bool(cached_tokens),
        "shard_index": shard_index, "shard_count": shard_count, "architecture": architecture}
    manifest["image_ids_path"] = str(image_ids_path) if image_ids_path else None
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _load_cached_tokens(path: str | Path | None) -> dict[str, list[OCRToken]]:
    cached: dict[str, list[OCRToken]] = {}
    if not path or not Path(path).exists():
        return cached
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line); cached[row["image_id"]] = [OCRToken(**item) for item in row.get("tokens", [])]
    return cached


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def _metrics(predictions: list[dict[str, Any]], review_rows: list[dict[str, Any]], labels_path: str | Path | None,
             latencies: list[float], stage_rows: list[dict[str, float | None]], config: dict[str, Any],
             wall_seconds: float, peak_rss: int | None) -> dict[str, Any]:
    none_count = sum(item["final_date"] == "NONE" for item in predictions)
    result: dict[str, Any] = {"image_count": len(predictions), "none_count": none_count,
        "none_rate": none_count / len(predictions) if predictions else 0.0,
        "latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        "latency_ms_median": statistics.median(latencies) if latencies else 0.0,
        "latency_ms_p95": _percentile(latencies, .95), "wall_seconds": wall_seconds,
        "sec_per_image": wall_seconds / len(predictions) if predictions else 0.0,
        "images_per_sec": len(predictions) / wall_seconds if wall_seconds and predictions else 0.0,
        "peak_rss_bytes": peak_rss, "peak_ram_mb": peak_rss / (1024 * 1024) if peak_rss else None}
    weight_bytes = _model_weight_bytes(config)
    result["model_weight_bytes"], result["model_weight_mb"] = weight_bytes, weight_bytes / (1024 * 1024) if weight_bytes else None
    for key in ("preprocess_ms", "detection_ms", "recognition_ms", "ocr_ms", "candidate_generation_ms", "selection_ms", "normalization_ms"):
        values = [float(row[key]) for row in stage_rows if row.get(key) is not None]
        result[f"{key}_mean"], result[f"{key}_p95"] = (statistics.mean(values) if values else None), (_percentile(values, .95) if values else None)
    result["detection_recognition_timing_available"] = any(row.get("detection_ms") is not None and row.get("recognition_ms") is not None for row in stage_rows)
    labels = _load_labels(labels_path)
    comparable = [item for item in predictions if item["image_id"] in labels]
    if not labels_path:
        _set_error_categories(review_rows, labels); result["recognition_metrics"] = {"status": "unavailable", "reason": "labels_path_not_provided"}; result["error_categories"] = _error_category_counts(review_rows); return result
    result["labeled_count"] = len(comparable)
    result["final_date_exact_match"] = sum(_prediction_matches(item["final_date"], labels[item["image_id"]]["final_date"]) for item in comparable) / len(comparable) if comparable else 0.0
    non_none = [item for item in comparable if parse_final_date(labels[item["image_id"]]["final_date"]) is not None]
    recall_hits = selection_hits = year_hits = month_hits = day_hits = 0
    for item in non_none:
        row = next(r for r in review_rows if r["image_id"] == item["image_id"]); target = parse_final_date(labels[item["image_id"]]["final_date"])
        candidate_keys = [parse_final_date(value) for value in str(row.get("candidate_set", "")).split(" | ") if value]
        row["candidate_recall"] = target in candidate_keys; recall_hits += int(row["candidate_recall"])
        selection_hits += int(row["candidate_recall"] and _prediction_matches(item["final_date"], labels[item["image_id"]]["final_date"]))
        predicted = parse_final_date(item["final_date"])
        if predicted and target:
            year_hits += int(predicted[0] == target[0]); month_hits += int(predicted[1] == target[1]); day_hits += int(predicted[2] == target[2])
    denominator = len(non_none); recalled = sum(1 for row in review_rows if row.get("candidate_recall") is True)
    result.update({"candidate_recall_count": recall_hits, "candidate_recall_denominator": denominator,
        "candidate_recall": recall_hits / denominator if denominator else None,
        "candidate_selection_accuracy": selection_hits / recalled if recalled else None,
        "candidate_selection_denominator": recalled, "year_accuracy": year_hits / denominator if denominator else None,
        "month_accuracy": month_hits / denominator if denominator else None, "day_accuracy": day_hits / denominator if denominator else None,
        "recognition_metrics": _recognition_metrics(labels, review_rows)})
    _set_error_categories(review_rows, labels); result["error_categories"] = _error_category_counts(review_rows)
    return result


def _load_labels(path: str | Path | None) -> dict[str, dict[str, str]]:
    if not path: return {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {str(row.get("image_id", "")): {str(k): str(v) for k, v in row.items() if v is not None} for row in csv.DictReader(handle)}


def _prediction_matches(predicted: str, target: str) -> bool:
    return predicted == target or parse_final_date(predicted) == parse_final_date(target)


def _set_error_categories(rows: list[dict[str, Any]], labels: dict[str, dict[str, str]]) -> None:
    for row in rows:
        label = labels.get(row["image_id"])
        if not label or _prediction_matches(str(row["final_date"]), str(label.get("final_date", "NONE"))): row["error_category"] = ""; continue
        target = parse_final_date(label.get("final_date", "NONE")); values = [parse_final_date(v) for v in str(row.get("candidate_set", "")).split(" | ") if v]
        if target is None:
            category = "AMBIGUOUS / UNKNOWN"
        elif target in values: category = "SEL"
        elif not row.get("token_count"): category = "DET"
        elif _DATEISH.search(str(row.get("ocr_text", ""))): category = "GEN"
        elif target is not None: category = "REC"
        else: category = "AMBIGUOUS / UNKNOWN"
        if target is not None and parse_final_date(str(row.get("final_date", "NONE"))) == target: category = "NORM"
        row["error_category"] = category


def _error_category_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get("error_category")
        if key:
            counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    return {key: {"count": value, "ratio": value / total if total else 0.0} for key, value in sorted(counts.items())}


def _recognition_metrics(labels: dict[str, dict[str, str]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [(row, labels[row["image_id"]].get("ocr_text") or labels[row["image_id"]].get("text")) for row in rows if row["image_id"] in labels]
    pairs = [(row, text) for row, text in pairs if text]
    if not pairs: return {"status": "unavailable", "reason": "labels_have_no_ocr_text_or_text_column"}
    distance = sum(_levenshtein(str(row.get("ocr_text", "")), str(text)) for row, text in pairs); length = sum(max(1, len(str(text))) for _, text in pairs)
    return {"status": "available", "cer": distance / length, "exact_string_accuracy": sum(str(row.get("ocr_text", "")) == str(text) for row, text in pairs) / len(pairs), "labeled_count": len(pairs)}


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1): current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _percentile(values: list[float], quantile: float) -> float:
    if not values: return 0.0
    ordered = sorted(values); return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * quantile) - 1))]


def _rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _model_weight_bytes(config: dict[str, Any]) -> int | None:
    _, params = component_spec(config, "ocr", "mock"); paths: set[Path] = set()
    for key, value in params.items():
        if isinstance(value, str) and key.endswith(("_dir", "_path")):
            path = Path(os.path.expandvars(value))
            if path.exists(): paths.add(path)
    if not paths: return None
    return sum(item.stat().st_size for path in paths for item in (path.rglob("*") if path.is_dir() else [path]) if item.is_file())
