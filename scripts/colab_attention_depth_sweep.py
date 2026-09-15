"""Run a GPU-only synthetic SVTR capacity sweep without Drive access."""
import importlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

extract_root = Path('/content/itda_attention')
extract_root.mkdir(exist_ok=True)
with tarfile.open('/content/itda_attention_bundle.tar.gz') as archive:
    archive.extractall(extract_root)
root = extract_root / 'itda_ocr'
os.chdir(root)
sys.path.insert(0, str(root / 'src'))
importlib.invalidate_caches()

variants = [
    ('svtr_base_d96_l2', 96, 4, 2),
    ('svtr_deep_d96_l4', 96, 4, 4),
    ('svtr_wide_d128_l2', 128, 4, 2),
    ('svtr_deepwide_d160_l4', 160, 8, 4),
]
summary = []
for name, d_model, heads, layers in variants:
    out = Path('/content/attention_depth_sweep') / name
    command = [sys.executable, 'scripts/train_synthetic_svtr_tiny.py',
               '--output', str(out), '--train-count', '10000', '--val-count', '1000',
               '--epochs', '12', '--batch-size', '128', '--device', 'cuda', '--num-workers', '2',
               '--d-model', str(d_model), '--heads', str(heads), '--layers', str(layers)]
    subprocess.run(command, check=True)
    history = json.loads((out / 'history.json').read_text())
    best = max(history, key=lambda row: row['val_exact'])
    summary.append({'name': name, 'd_model': d_model, 'heads': heads, 'layers': layers,
                    'best_epoch': best['epoch'], 'best_exact': best['val_exact'], 'best_cer': best['val_cer']})
(Path('/content/attention_depth_sweep') / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
