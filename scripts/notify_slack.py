#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or send an experiment summary to Slack.")
    parser.add_argument("run_dir", nargs="?", help="Single-run artifact directory")
    parser.add_argument("--gpu-run")
    parser.add_argument("--cpu-run")
    parser.add_argument("--artifact-url")
    parser.add_argument("--github-url")
    parser.add_argument("--send", action="store_true", help="Send via SLACK_WEBHOOK_URL")
    args = parser.parse_args()
    cpu_dir = Path(args.cpu_run) if args.cpu_run else None
    gpu_dir = Path(args.gpu_run) if args.gpu_run else None
    if not (args.run_dir or args.cpu_run or args.gpu_run):
        raise SystemExit("Provide run_dir or --gpu-run/--cpu-run")
    run_dir = cpu_dir or gpu_dir or Path(args.run_dir)
    metrics = _read_json(run_dir / "metrics.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    architecture = _read_json(run_dir / "architecture.json")
    gpu_metrics = _read_json(gpu_dir / "metrics.json") if gpu_dir and (gpu_dir / "metrics.json").exists() else None
    text = _format(metrics, manifest, architecture, args.artifact_url, args.github_url, gpu_metrics)
    payload = {"text": text}
    (run_dir / "slack_message.md").write_text(text + "\n", encoding="utf-8")
    if args.send:
        webhook = os.environ.get("SLACK_WEBHOOK_URL")
        if not webhook:
            raise SystemExit("--send requires SLACK_WEBHOOK_URL")
        request = urllib.request.Request(
            webhook,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status >= 300:
                raise SystemExit(f"Slack webhook failed: HTTP {response.status}")
    print(text)
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format(metrics: dict[str, Any], manifest: dict[str, Any], architecture: dict[str, Any], artifact_url: str | None, github_url: str | None, gpu_metrics: dict[str, Any] | None = None) -> str:
    runtime = architecture.get("runtime", {})
    lines = [
        f"*OCR experiment {'complete'}*",
        f"experiment: `{manifest.get('config', {}).get('name', 'unnamed')}`",
        f"device: `{manifest.get('device', runtime.get('device', 'unknown'))}`",
        f"images: `{manifest.get('image_count', 'unknown')}`",
        "",
        "*Architecture*",
        f"• preprocess: `{architecture.get('preprocess', {}).get('plugin', 'none')}`",
        f"  params: `{_params_text(architecture.get('preprocess', {}))}`",
        f"• OCR: `{architecture.get('ocr', {}).get('plugin', 'none')}`",
        f"  params: `{_params_text(architecture.get('ocr', {}))}`",
        f"• selector: `{architecture.get('selector', {}).get('plugin', 'none')}`",
        f"  params: `{_params_text(architecture.get('selector', {}))}`",
        f"• normalizer: `{architecture.get('normalizer', {}).get('plugin', 'none')}`",
        f"  params: `{_params_text(architecture.get('normalizer', {}))}`",
        f"• runtime: `{runtime.get('device', 'unknown')}`, threads=`{runtime.get('threads', 'default')}`",
        "",
        "*Metrics*",
        f"• CPU exact-match: `{metrics.get('final_date_exact_match', 'n/a')}`",
        f"• latency mean: `{metrics.get('latency_ms_mean', 'n/a')} ms`",
        f"• latency p95: `{metrics.get('latency_ms_p95', 'n/a')} ms`",
        f"• NONE rate: `{metrics.get('none_rate', 'n/a')}`",
    ]
    weight_id = architecture.get("ocr", {}).get("weight_id")
    if weight_id:
        lines.insert(lines.index("• selector: `" + str(architecture.get('selector', {}).get('plugin', 'none')) + "`"), f"• OCR weight: `{weight_id}`")
    if gpu_metrics is not None:
        lines.insert(lines.index("*Metrics*") + 1, f"• GPU exact-match: `{gpu_metrics.get('final_date_exact_match', 'n/a')}`")
    if artifact_url:
        lines.append(f"• artifacts: {artifact_url}")
    if github_url:
        lines.append(f"• GitHub: {github_url}")
    return "\n".join(lines)


def _params_text(component: dict[str, Any]) -> str:
    params = component.get("params", {}) or {}
    if not params:
        return "none"
    text = json.dumps(params, ensure_ascii=False, separators=(", ", ":"))
    return text if len(text) <= 500 else text[:497] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
