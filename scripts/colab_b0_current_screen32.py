import importlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
import yaml

extract_root = Path('/content/itda_ocr_current')
extract_root.mkdir(exist_ok=True)
with tarfile.open('/content/itda_ocr_bundle_current.tar.gz') as archive:
    archive.extractall(extract_root)
root = extract_root / 'itda_ocr'
os.chdir(root)
sys.path.insert(0, str(root / 'src'))
importlib.invalidate_caches()
config_path = root / 'configs/experiments/b0_pp_ocr_mobile_raw_rule.yaml'
config = yaml.safe_load(config_path.read_text())
config['ocr']['params']['enable_mkldnn'] = False
config['ocr']['params']['fallback_enabled'] = False
config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
weights = Path('/content/drive/MyDrive/상품사진입니다/weights')
env = os.environ.copy(); env['ITDA_WEIGHTS_ROOT'] = str(weights)
# Paddle 3.3.1's PIR mobile-det graph fails in the oneDNN executor on this
# CPU VM. Keep the production config's oneDNN setting, but disable the broken
# runtime flag for this compatibility rerun.
env['FLAGS_use_mkldnn'] = '0'
result = subprocess.run([
    sys.executable, 'scripts/run_experiment.py',
    '--config', 'configs/experiments/b0_pp_ocr_mobile_raw_rule.yaml',
    '--input', '/content/drive/MyDrive/상품사진입니다',
    '--output', '/content/B0_CURRENT_screen32',
    '--labels', '/content/labels.csv',
    '--image-ids-file', '/content/screen32_ids.txt',
    '--device', 'cpu', '--threads', '4',
], check=False, env=env, capture_output=True, text=True)
print('returncode', result.returncode)
print(result.stdout[-10000:])
print(result.stderr[-10000:])
if result.returncode:
    raise SystemExit(result.returncode)
print(json.loads(Path('/content/B0_CURRENT_screen32/metrics.json').read_text()))
