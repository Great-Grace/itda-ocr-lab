from pathlib import Path
import tarfile
root = Path('/content/attention_depth_sweep')
with tarfile.open('/content/attention_depth_sweep_artifacts.tar.gz', 'w:gz') as archive:
    archive.add(root / 'summary.json', arcname='summary.json')
    for path in root.glob('*/history.json'):
        archive.add(path, arcname=str(path.relative_to(root)))
    for path in root.glob('*/model.json'):
        archive.add(path, arcname=str(path.relative_to(root)))
print('packed')
