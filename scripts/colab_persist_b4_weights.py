from pathlib import Path
import shutil

source = Path('/root/.paddlex/official_models/PP-OCRv6_medium_rec')
target = Path('/content/drive/MyDrive/상품사진입니다/weights/paddle/ppocrv6_medium_rec')
if not source.is_dir():
    raise RuntimeError(f'missing model cache: {source}')
if target.exists():
    shutil.rmtree(target)
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(source, target)
size = sum(path.stat().st_size for path in target.rglob('*') if path.is_file())
print(f'persisted {target} bytes={size}')
