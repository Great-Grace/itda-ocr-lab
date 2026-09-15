from pathlib import Path
root=Path('/content/drive/MyDrive/상품사진입니다')
for p in sorted(root.glob('weights/paddle/*')):
    print(p, p.is_dir(), len(list(p.rglob('*'))) if p.is_dir() else p.stat().st_size)
