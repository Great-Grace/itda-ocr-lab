#!/usr/bin/env python3
"""Create an isolated experiment branch and config from the common template."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start one isolated OCR experiment.")
    parser.add_argument("--name", required=True, help="Short experiment slug, e.g. msr-resize-v1")
    parser.add_argument("--owner", required=True, help="Team member handle used in the branch name")
    parser.add_argument("--base", default="main")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not SLUG.fullmatch(args.name) or not SLUG.fullmatch(args.owner):
        raise SystemExit("--name and --owner must use lowercase letters, digits, and hyphens only")
    branch = f"feature/{args.owner}-{args.name}"
    config_path = ROOT / "configs" / "experiments" / f"{args.name}.yaml"
    if config_path.exists():
        raise SystemExit(f"Experiment config already exists: {config_path}")
    if args.dry_run:
        print(f"Would create branch {branch} from {args.base} and config {config_path.relative_to(ROOT)}")
        return 0
    if _git("status", "--porcelain").strip():
        raise SystemExit("Working tree is not clean. Commit or stash current work before starting an isolated experiment.")
    _git("fetch", "origin", args.base)
    _git("switch", "--create", branch, f"origin/{args.base}")
    template = yaml.safe_load((ROOT / "configs" / "templates" / "experiment.yaml").read_text(encoding="utf-8"))
    template["name"] = args.name
    config_path.write_text(yaml.safe_dump(template, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Ready: branch={branch}; config={config_path.relative_to(ROOT)}")
    return 0


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout


if __name__ == "__main__":
    raise SystemExit(main())
