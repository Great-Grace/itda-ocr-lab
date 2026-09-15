
import os
from pathlib import Path

out_dir = Path("/content/itda_ocr/runs/measured/smoke_val_10_smoke")
if out_dir.exists():
    print("Files in smoke_val_10_smoke:", os.listdir(str(out_dir)))
    stdout = out_dir / "runner.stdout.log"
    if stdout.exists():
        print("--- stdout ---")
        print(stdout.read_text()[-500:])
    stderr = out_dir / "runner.stderr.log"
    if stderr.exists():
        print("--- stderr ---")
        print(stderr.read_text()[-500:])
else:
    print("smoke_val_10_smoke dir does not exist yet")
