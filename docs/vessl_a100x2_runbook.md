# VESSL A100 x2 runbook

## Console selections

- A100 SXM, x2, a region that shows high availability
- 100GB persistent volume, mounted at `/data`, in the same region
- Managed Torch CUDA 13 / Python 3.13 image
- SSH key `DAT-OCR`

## Local, once

```bash
./scripts/package_vessl_bundle.sh
vesslctl volume upload <volume-slug> runs/vessl/itda_ocr_code_bundle.tar.gz
```

This uploads only code and approved split metadata. It does not upload images or weights.

## SSH, once

```bash
curl -fsSL https://rclone.org/install.sh | sudo bash
rclone config
export DATA_ROOT=/data/itda
export RCLONE_REMOTE=gdrive
export DRIVE_DATASET_PATH='상품사진입니다'
export BUNDLE_PATH="$(find /data -name itda_ocr_code_bundle.tar.gz -print -quit)"
mkdir -p /workspace/itda-ocr
tar -xzf "$BUNDLE_PATH" -C /workspace/itda-ocr
bash /workspace/itda-ocr/scripts/vessl_bootstrap_a100x2.sh
```

The interactive `rclone config` OAuth step is the only manual Google Drive authorization. The bootstrap verifies every split filename against the synced image directory before GPU work begins.

## GPU lanes after bootstrap

Use independent processes, not DDP:

```bash
tmux new -s itda-gpu
CUDA_VISIBLE_DEVICES=0 <crop-harvesting-command>
CUDA_VISIBLE_DEVICES=1 <real-crop-training-command>
```

After harvesting, split independent architectures over the two GPUs. Save every checkpoint, crop manifest, config hash, and validation history under `/data/itda/artifacts/`.

Do not run final benchmark inference on either GPU. Submit measured CPU runs separately after GPU candidate selection.
