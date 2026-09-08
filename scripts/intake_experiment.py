#!/usr/bin/env python3
"""Validate an agent-produced experiment request and list missing decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "baseline_config": "configs/baseline_mock.yaml",
    "metric": "final_date_exact_match",
    "gpu_type": "T4",
    "max_gpu_hours": 2,
    "cpu_required": True,
}

REQUIRED = {
    "hypothesis": "논문 또는 아이디어를 이번 실험에서 무엇으로 검증하나요?",
    "target": "어느 모듈을 바꾸나요? preprocess / ocr / selector / normalizer 중 하나를 지정해 주세요.",
    "input_dir": "실험 이미지 경로는 어디인가요?",
    "output_name": "실험 결과 이름은 무엇인가요?",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check an experiment request before expensive execution.")
    parser.add_argument("request", help="YAML or JSON experiment request")
    parser.add_argument("--output", help="Write the resolved plan here")
    args = parser.parse_args()

    request = _load(Path(args.request))
    resolved = {**DEFAULTS, **request}
    missing = [key for key in REQUIRED if not resolved.get(key)]
    questions = [REQUIRED[key] for key in missing]
    result = {
        "status": "needs_input" if missing else "ready",
        "missing_fields": missing,
        "questions": questions,
        "resolved_plan": resolved,
    }
    if args.output and not missing:
        Path(args.output).write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
        result["plan_path"] = args.output
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not missing else 2


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit("Experiment request must be a YAML/JSON mapping")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
