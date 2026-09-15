from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path


root = Path("/content/itda_measured_val")
root.mkdir(parents=True, exist_ok=True)
with tarfile.open("/content/measured_val_payload.tar.gz", "r:gz") as archive:
    archive.extractall(root)
subprocess.run([sys.executable, "scripts/colab_measured_val_worker.py"], cwd=root, check=True)
