from pathlib import Path
root = Path("/content/drive/MyDrive/상품사진입니다/weights")
for path in sorted(root.rglob("*")):
    if path.is_file() and ("v6" in path.name.lower() or "ocr" in path.name.lower() or path.name == "inference.yml"):
        print(path, path.stat().st_size)
