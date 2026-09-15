from pathlib import Path
import tarfile
with tarfile.open('/content/svtr_roi_artifacts.tar.gz', 'w:gz') as archive:
    for path in (Path('/content/svtr_merged_tokens.jsonl'), Path('/content/B9_roi_runtime.json'), Path('/content/B9_SVTR_ROI_SECOND_PASS_screen32')):
        if path.exists():
            archive.add(path, arcname=path.name)
print('packed', Path('/content/svtr_roi_artifacts.tar.gz').stat().st_size)
