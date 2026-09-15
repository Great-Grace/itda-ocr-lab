#!/usr/bin/env bash
set -euo pipefail

# Run inside the VESSL A100 x2 workspace after the code bundle is copied to
# /data/itda/bootstrap/. Do not point RCLONE_REMOTE at a public URL: rclone
# must use the user's authenticated Google Drive remote.
DATA_ROOT="${DATA_ROOT:-/data/itda}"
WORK_ROOT="${WORK_ROOT:-/workspace/itda-ocr}"
VENV_ROOT="${VENV_ROOT:-$DATA_ROOT/venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
DRIVE_DATASET_PATH="${DRIVE_DATASET_PATH:-상품사진입니다}"
BUNDLE_PATH="${BUNDLE_PATH:-$DATA_ROOT/bootstrap/itda_ocr_code_bundle.tar.gz}"

mkdir -p "$DATA_ROOT" "$WORK_ROOT"
test -f "$BUNDLE_PATH" || { echo "Missing code bundle: $BUNDLE_PATH"; exit 1; }
tar -xzf "$BUNDLE_PATH" -C "$WORK_ROOT"
cd "$WORK_ROOT"

"$PYTHON_BIN" -m venv --system-site-packages "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
python -m pip install paddleocr==3.7.0 opencv-python-headless

mkdir -p "$DATA_ROOT/images" "$DATA_ROOT/weights" "$DATA_ROOT/artifacts" "$DATA_ROOT/bootstrap"
if [ "${SKIP_DRIVE_SYNC:-0}" != "1" ]; then
  command -v rclone >/dev/null || { echo "Install rclone, then configure an authenticated '$RCLONE_REMOTE:' remote."; exit 1; }
  rclone lsd "$RCLONE_REMOTE:" >/dev/null
  rclone copy "$RCLONE_REMOTE:$DRIVE_DATASET_PATH" "$DATA_ROOT/images" \
    --include '*.jpg' --include '*.jpeg' --include '*.png' --include '*.webp' \
    --transfers 16 --checkers 32 --fast-list --progress
  rclone copy "$RCLONE_REMOTE:$DRIVE_DATASET_PATH/weights" "$DATA_ROOT/weights" \
    --transfers 8 --checkers 16 --fast-list --progress
else
  echo "SKIP_DRIVE_SYNC=1: reusing verified /data images and weights"
fi

export ITDA_WEIGHTS_ROOT="$DATA_ROOT/weights"
export ITDA_V6_WEIGHTS_ROOT="$DATA_ROOT/weights/PP-OCRv6_medium_rec"
if [ ! -d "$ITDA_V6_WEIGHTS_ROOT" ] && [ -f "$DATA_ROOT/weights/ppocrv6_medium_rec.tar.gz" ]; then
  mkdir -p "$DATA_ROOT/weights/v6_extract"
  tar -xzf "$DATA_ROOT/weights/ppocrv6_medium_rec.tar.gz" -C "$DATA_ROOT/weights/v6_extract"
  ITDA_V6_WEIGHTS_ROOT="$(find "$DATA_ROOT/weights/v6_extract" -name inference.yml -print -quit | xargs -n1 dirname)"
  export ITDA_V6_WEIGHTS_ROOT
fi

python - <<'PY'
from pathlib import Path
import csv, os
root = Path(os.environ['DATA_ROOT']) if 'DATA_ROOT' in os.environ else Path('/data/itda')
images = {p.stem for p in (root / 'images').iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}}
for split in ('train', 'val', 'test'):
    rows = list(csv.DictReader((Path.cwd() / 'data/splits' / f'{split}.csv').open(encoding='utf-8')))
    missing = [Path(row['filename']).stem for row in rows if Path(row['filename']).stem not in images]
    print(split, 'rows=', len(rows), 'missing_images=', len(missing))
    if missing: raise SystemExit(f'{split} has missing images: {missing[:10]}')
print('images=', len(images))
PY

python - <<'PY'
import paddle, torch
print('torch_cuda=', torch.cuda.is_available(), 'count=', torch.cuda.device_count())
print('paddle_cuda=', paddle.device.is_compiled_with_cuda(), 'device=', paddle.device.get_device())
PY

echo "BOOTSTRAP_OK"
