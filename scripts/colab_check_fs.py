
import os, sys
from pathlib import Path

print("Python:", sys.version.split()[0])
print("/content:", os.listdir("/content"))
drive_p = Path("/content/drive/MyDrive/상품사진입니다")
print("Drive 상품사진입니다 exists:", drive_p.exists())
if drive_p.exists():
    imgs = [x for x in drive_p.iterdir() if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print("Found images in Drive:", len(imgs))
    weights = drive_p / "weights"
    print("Weights exists:", weights.exists())
    if weights.exists():
        print("Weights content:", os.listdir(str(weights)))
