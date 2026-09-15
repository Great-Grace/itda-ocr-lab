from __future__ import annotations
import os
import subprocess
import sys

subprocess.run(['bash','-lc','mkdir -p /content/itda_ocr_lab && tar -xzf /content/itda_ocr_bundle.tar.gz -C /content/itda_ocr_lab'], check=True)
os.chdir('/content/itda_ocr_lab')
subprocess.run([
    sys.executable, 'scripts/train_synthetic_svtr_tiny.py',
    '--output','/content/attn_runs/svtr_tiny_ctc_long',
    '--train-count','20000','--val-count','2000','--epochs','20','--batch-size','128','--device','cuda'
], check=True)
