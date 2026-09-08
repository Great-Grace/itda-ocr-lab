# Weight registry

The registry is the only approved way for an agent to choose model weights. Add an entry to `configs/weights.yaml` after checking the source, license, task, language, local path, and SHA-256 checksum.

```yaml
weights:
  paddle_ocr_ko_mobile:
    task: detection_and_recognition
    language: ko_en_numeric
    source: approved_release_url
    license: Apache-2.0
    local_path: weights/paddle_ocr_ko_mobile
    sha256: replace_me
    supports_finetuning: false
    cpu_profile: light
```

Agents may propose a new entry, but the entry must be reviewed before a GPU run. `predict.ipynb` must never download weights at inference time.
