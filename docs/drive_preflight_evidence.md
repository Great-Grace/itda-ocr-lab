# Drive preflight evidence

The connected Google Drive provider currently resolves the dataset folder
`상품사진입니다` as:

- Folder ID: `1oiilNn_TusjjtUpbCdG2ihrjmRz-ffKJ`
- Manifest: `DATASET_MANIFEST.yaml`
- Expected image count: `3352`
- Allowed extensions: `bmp`, `jpeg`, `jpg`, `png`, `webp`
- Weights folder ID: `1bGHVdQO0xWq99s0Y4I9fJQ8hv5gobRLg`
- PP-OCRv6 archive present: `ppocrv6_medium_rec.tar.gz` (67,475,489 bytes)
- Paddle weight subfolder present: `paddle`

This proves Drive-side access and dataset metadata. It does not mean a Colab
runtime is currently mounted. The Colab routine must still run the readonly
mount and local preflight before an OCR job starts.

For a mount-independent smoke of the real data, 32 `screen32` images and the
PP-OCRv5 detector/recognizer files were fetched through the Drive connector
into `/private/tmp/itda_drive_images` and `/private/tmp/itda_drive_weights`.
The resulting CPU run is `runs/B0_LOCAL_DRIVE_SCREEN32` and passes the
submission schema check.
