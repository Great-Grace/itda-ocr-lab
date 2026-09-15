from pathlib import Path
import tarfile
with tarfile.open('/content/b0_current_artifacts.tar.gz', 'w:gz') as archive:
    archive.add('/content/B0_CURRENT_screen32', arcname='B0_CURRENT_screen32')
print('packed')
