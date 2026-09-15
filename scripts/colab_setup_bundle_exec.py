import shutil, tarfile
from pathlib import Path
root=Path('/content/itda_lab')
if root.exists(): shutil.rmtree(root)
root.mkdir()
with tarfile.open('/content/itda_ocr_code_bundle.tar.gz','r:gz') as archive:
    archive.extractall(root)
print('bundle_ready',root)
print('val_exists',(root/'data/splits/val.csv').exists())
