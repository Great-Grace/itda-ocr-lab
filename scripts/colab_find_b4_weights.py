from pathlib import Path
for root in (Path('/root/.paddlex'), Path('/root/.cache'), Path('/content')):
    if not root.exists():
        continue
    for path in root.rglob('*PP-OCRv6*'):
        print(path)
