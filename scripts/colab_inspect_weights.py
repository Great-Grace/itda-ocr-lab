
import os, tarfile
from pathlib import Path

drive = Path("/content/drive/MyDrive/상품사진입니다/weights")
print("Weights path exists:", drive.exists())
paddle_dir = drive / "paddle"
if paddle_dir.exists():
    print("paddle subdirs:", os.listdir(str(paddle_dir)))

tar_path = drive / "ppocrv6_medium_rec.tar.gz"
if tar_path.is_file():
    with tarfile.open(tar_path, "r:gz") as tar:
        print("tar members:", [m.name for m in tar.getmembers()[:10]])
