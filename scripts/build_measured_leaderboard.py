#!/usr/bin/env python3
"""Build a leaderboard only from hash-verified measured experiment evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(run: Path, evidence: dict[str, Any]) -> str | None:
    if evidence.get("status") != "complete":
        return "run_not_complete"
    if not evidence.get("rank_eligible"):
        return str(evidence.get("ranking_disallowed_reason") or "not_rank_eligible")
    required = {
        "metrics_sha256": "metrics.json",
        "predictions_sha256": "predictions.csv",
        "tokens_sha256": "ocr_tokens.jsonl",
    }
    for evidence_key, filename in required.items():
        path = run / filename
        if not path.is_file() or evidence.get(evidence_key) != sha256_file(path):
            return f"artifact_hash_mismatch:{filename}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs/measured")
    parser.add_argument("--output", default="runs/measured/leaderboard.json")
    args = parser.parse_args()
    runs_root = (ROOT / args.runs_root).resolve()
    accepted, rejected = [], []
    for evidence_path in sorted(runs_root.glob("*/evidence.json")):
        run = evidence_path.parent
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except Exception as exc:
            rejected.append({"run": str(run.relative_to(ROOT)), "reason": f"invalid_evidence:{exc}"})
            continue
        reason = verify(run, evidence)
        if reason:
            rejected.append({"run": str(run.relative_to(ROOT)), "reason": reason})
            continue
        metrics = evidence.get("metrics") or {}
        accepted.append({
            "name": evidence.get("name"),
            "run": str(run.relative_to(ROOT)),
            "label_provenance": evidence.get("label_provenance"),
            "ranking_scope": evidence.get("ranking_scope"),
            "config": evidence.get("config"),
            "image_count": evidence.get("image_count"),
            **metrics,
        })
    accepted.sort(key=lambda row: (-(row.get("final_date_exact_match") or -1), row.get("sec_per_image") or float("inf"), row["name"] or ""))
    for rank, row in enumerate(accepted, start=1):
        row["rank"] = rank
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"accepted": accepted, "rejected": rejected}, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = ["# Measured OCR leaderboard", "", "Only fresh OCR runs with complete hash-verified evidence are ranked.", "",
                "| Rank | Experiment | Label provenance | EM | Candidate recall | Selection acc | sec/image | Peak RAM |",
                "|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in accepted:
        def pct(value: Any) -> str:
            return "N/A" if value is None else f"{float(value) * 100:.2f}%"
        def num(value: Any) -> str:
            return "N/A" if value is None else f"{float(value):.3f}"
        markdown.append(f"| {row['rank']} | {row['name']} | {row['label_provenance']} | {pct(row.get('final_date_exact_match'))} | {pct(row.get('candidate_recall'))} | {pct(row.get('candidate_selection_accuracy'))} | {num(row.get('sec_per_image'))} | {num(row.get('peak_ram_mb'))} |")
    markdown.extend(["", "## Rejected runs", ""])
    if rejected:
        markdown.extend(f"- `{row['run']}`: {row['reason']}" for row in rejected)
    else:
        markdown.append("- None")
    output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
