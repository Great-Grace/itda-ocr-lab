#!/usr/bin/env python3
"""Sync a branch/config to Colab, run GPU+CPU jobs, and collect artifacts.

The script is intentionally explicit about the data source: images and labels
are expected to be available under the mounted shared Drive folder. Code and
config are bundled from the current Git checkout; weights stay in Drive or a
pre-approved local path and are never downloaded by inference.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_lab.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an OCR experiment on Colab or locally.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["plan", "local", "colab"], default="plan")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--dataset-root", help="Mounted Drive dataset root; auto-discovery is used when omitted")
    parser.add_argument("--mount-root", default="/content/drive")
    parser.add_argument("--labels", help="Labels path relative to dataset root")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--keep", action="store_true", help="Keep the Colab session for debugging")
    parser.add_argument("--notify-slack", action="store_true")
    parser.add_argument("--send-slack", action="store_true")
    parser.add_argument("--artifact-url")
    parser.add_argument("--github-url")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = _resolve_dataset(args.dataset_root, args.mount_root)
    plan = _plan(config, args, dataset)
    if args.mode == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "local":
        return _run_local(args, config, dataset)
    return _run_colab(args, config, dataset, plan)


def _resolve_dataset(dataset_root: str | None, mount_root: str) -> dict[str, str]:
    if dataset_root:
        root = Path(dataset_root)
    else:
        candidates = [
            Path(mount_root) / "MyDrive/ITDA_OCR",
            Path(mount_root) / "Shareddrives/ITDA_OCR",
            Path(mount_root) / "MyDrive/ITDA_OCR_DATASET",
        ]
        roots = [candidate for candidate in candidates if (candidate / "DATASET_MANIFEST.yaml").exists()]
        if len(roots) != 1:
            raise SystemExit("Dataset was not uniquely resolved. Pass --dataset-root to a folder containing DATASET_MANIFEST.yaml.")
        root = roots[0]
    manifest_path = root / "DATASET_MANIFEST.yaml"
    if not manifest_path.exists():
        raise SystemExit(f"Missing dataset manifest: {manifest_path}")
    manifest = _load_yaml(manifest_path)
    image_dir = root / str(manifest.get("image_dir", "images"))
    expected = int(manifest.get("expected_image_count", 3352))
    image_count = sum(1 for path in image_dir.iterdir() if path.is_file()) if image_dir.exists() else 0
    if image_count != expected:
        raise SystemExit(f"Dataset count mismatch: expected {expected}, found {image_count} in {image_dir}")
    return {
        "root": str(root),
        "image_dir": str(image_dir),
        "manifest": str(manifest_path),
        "dataset_id": str(manifest.get("dataset_id", "unknown")),
        "version": str(manifest.get("version", "unknown")),
    }


def _plan(config: dict, args: argparse.Namespace, dataset: dict[str, str]) -> dict:
    return {
        "mode": args.mode,
        "gpu": args.gpu,
        "config": str(Path(args.config).resolve()),
        "output": str(Path(args.output).resolve()),
        "dataset": dataset,
        "labels": args.labels,
        "max_images": args.max_images,
        "notify_slack": args.notify_slack,
        "architecture": {
            name: config.get(name, {"plugin": "none", "params": {}})
            for name in ("preprocess", "ocr", "selector", "normalizer")
        },
        "runtime": config.get("runtime", {}),
    }


def _run_local(args: argparse.Namespace, config: dict, dataset: dict[str, str]) -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_experiment.py"),
        "--config", args.config,
        "--input", dataset["image_dir"],
        "--output", args.output,
        "--device", "cpu",
    ]
    if args.labels:
        command.extend(["--labels", str(Path(dataset["root"]) / args.labels)])
    if args.max_images:
        command.extend(["--max-images", str(args.max_images)])
    subprocess.run(command, check=True)
    if args.notify_slack:
        _notify(args)
    return 0


def _run_colab(args: argparse.Namespace, config: dict, dataset: dict[str, str], plan: dict) -> int:
    if shutil.which("colab") is None:
        raise SystemExit("Colab CLI is not installed. Install google-colab-cli and authenticate first.")
    session = f"itda-{Path(args.output).name}".replace("_", "-")[:40]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "source_bundle.tar.gz"
    remote_script = output_dir / "remote_run.py"
    _make_bundle(bundle_path, args.config)
    remote_script.write_text(_remote_script(dataset, args), encoding="utf-8")
    try:
        _run(["colab", "new", "-s", session, "--gpu", args.gpu])
        _run(["colab", "drivemount", "-s", session])
        _run(["colab", "upload", "-s", session, str(bundle_path), "/content/itda_ocr_bundle.tar.gz"])
        _run(["colab", "upload", "-s", session, str(remote_script), "/content/itda_remote_run.py"])
        _run(["colab", "exec", "-s", session, "-f", str(remote_script)])
        archive = output_dir / "colab_artifacts.tar.gz"
        _run(["colab", "download", "-s", session, "/content/itda_ocr_artifacts.tar.gz", str(archive)])
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(output_dir)
    finally:
        if not args.keep:
            _run(["colab", "stop", "-s", session], check=False)
    if args.notify_slack:
        _notify(args)
    return 0


def _make_bundle(destination: Path, config_path: str) -> None:
    with tarfile.open(destination, "w:gz") as tar:
        for relative in ("src", "scripts/run_experiment.py", "scripts/check_submission.py", "requirements.txt"):
            path = ROOT / relative
            if path.exists():
                tar.add(path, arcname=f"itda_ocr/{relative}")
        config = Path(config_path).resolve()
        tar.add(config, arcname=f"itda_ocr/{config.relative_to(ROOT)}")


def _remote_script(dataset: dict[str, str], args: argparse.Namespace) -> str:
    labels = str(Path(dataset["root"]) / args.labels) if args.labels else ""
    config_rel = Path(args.config).resolve().relative_to(ROOT)
    return f'''from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

bundle = Path("/content/itda_ocr_bundle.tar.gz")
with tarfile.open(bundle, "r:gz") as archive:
    archive.extractall("/content")
input_dir = {dataset["image_dir"]!r}
output_dir = Path("/content/itda_ocr/gpu_run")
output_dir.mkdir(parents=True, exist_ok=True)
command = [sys.executable, "/content/itda_ocr/scripts/run_experiment.py", "--config", "/content/itda_ocr/{config_rel}", "--input", input_dir, "--output", str(output_dir), "--device", "cuda"]
{f'command.extend(["--labels", {labels!r}])' if labels else ''}
{f'command.extend(["--max-images", "{args.max_images}"])' if args.max_images else ''}
subprocess.run(command, check=True)
cpu_dir = Path("/content/itda_ocr/cpu_run")
cpu_command = command[:]
cpu_command[cpu_command.index("--output") + 1] = str(cpu_dir)
cpu_command[cpu_command.index("--device") + 1] = "cpu"
subprocess.run(cpu_command, check=True)
with tarfile.open("/content/itda_ocr_artifacts.tar.gz", "w:gz") as archive:
    archive.add("/content/itda_ocr/gpu_run", arcname="gpu_run")
    archive.add("/content/itda_ocr/cpu_run", arcname="cpu_run")
'''


def _notify(args: argparse.Namespace) -> None:
    output = Path(args.output)
    command = [sys.executable, str(ROOT / "scripts/notify_slack.py")]
    if (output / "cpu_run").exists() or (output / "gpu_run").exists():
        command.extend(["--gpu-run", str(output / "gpu_run"), "--cpu-run", str(output / "cpu_run")])
    else:
        command.append(str(output))
    if args.send_slack:
        command.append("--send")
    if args.artifact_url:
        command.extend(["--artifact-url", args.artifact_url])
    if args.github_url:
        command.extend(["--github-url", args.github_url])
    subprocess.run(command, check=True)


def _run(command: list[str], check: bool = True) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=check)


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


if __name__ == "__main__":
    raise SystemExit(main())
