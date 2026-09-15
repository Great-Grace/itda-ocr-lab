"""Assemble the chunked final-test payload in the active Colab session."""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path


parts = sorted(Path("/content").glob("payload.part-*"))
if len(parts) != 13:
    raise RuntimeError(f"expected 13 payload chunks, found {len(parts)}")
archive_path = Path("/content/itda_final_payload.tar.gz")
with archive_path.open("wb") as destination:
    for part in parts:
        with part.open("rb") as source:
            shutil.copyfileobj(source, destination)

root = Path("/content/itda_final_payload")
if root.exists():
    shutil.rmtree(root)
root.mkdir()
with tarfile.open(archive_path, "r:gz") as archive:
    archive.extractall(root)

print({"payload_bytes": archive_path.stat().st_size, "root": str(root), "predict_exists": (root / "predict.ipynb").is_file()})
