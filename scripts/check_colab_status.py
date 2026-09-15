import subprocess
import os
from pathlib import Path

print("=== PS AUX (GEMINI) ===")
out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
for line in out.splitlines():
    if "gemini" in line:
        print(line)

print("\n=== TOTAL DRIVE IMAGES ===")
drive_dir = Path("/content/drive/MyDrive/상품사진입니다")
if drive_dir.exists():
    imgs = [p for p in drive_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    print(f"Total images found in Drive: {len(imgs)}")
else:
    print("Drive dir does not exist.")

print("\n=== CSV LINE COUNT & SAMPLES ===")
csv_path = Path("/content/runs/gemini_vlm_labels/gemini_labels.csv")
if csv_path.exists():
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    print(f"Total lines in gemini_labels.csv: {len(lines)}")
    print("Last 5 lines:")
    for l in lines[-5:]:
        print("  ", l)
else:
    print("CSV does not exist yet.")

print("\n=== DAEMON LOG (LAST 25 LINES) ===")
log_path = Path("/content/runs/gemini_vlm_labels/daemon.log")
if log_path.exists():
    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    print(f"Total log lines: {len(log_lines)}")
    for l in log_lines[-25:]:
        print(l)
else:
    print("daemon.log does not exist.")
