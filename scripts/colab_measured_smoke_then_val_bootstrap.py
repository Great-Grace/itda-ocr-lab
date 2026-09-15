import subprocess
import sys
import tarfile
from pathlib import Path

root = Path('/content/itda_measured_val_v2')
root.mkdir(parents=True, exist_ok=True)
with tarfile.open('/content/measured_val_payload_v2.tar.gz', 'r:gz') as archive:
    archive.extractall(root)
subprocess.run([sys.executable, 'scripts/colab_measured_smoke_then_val.py'], cwd=root, check=True)
