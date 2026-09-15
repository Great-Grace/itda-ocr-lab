
from pathlib import Path
p = Path("/content/drive/MyDrive/상품사진입니다")
print("MOUNT_CHECK:", p.exists())
if p.exists():
    imgs = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print("IMAGE_COUNT:", len(imgs))
    weights = p / "weights"
    print("WEIGHTS_EXIST:", weights.exists())
    if weights.exists():
        import os
        print("WEIGHTS_SUBDIRS:", os.listdir(str(weights)))
