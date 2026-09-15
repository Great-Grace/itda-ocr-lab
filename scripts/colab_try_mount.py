
import os, sys
from pathlib import Path
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("MOUNTED_OK")
except Exception as e:
    print("MOUNT_ERROR:", type(e), e)

p = Path("/content/drive/MyDrive/상품사진입니다")
print("Drive 상품사진입니다 exists:", p.exists())
if p.exists():
    print("Images count:", len([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg", ".png"}]))
