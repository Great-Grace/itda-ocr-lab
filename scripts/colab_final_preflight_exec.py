"""Argument-free Colab probe wrapper used by google-colab-cli exec."""
from pathlib import Path
import json
root = Path("/content/drive/MyDrive/상품사진입니다")
images = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
weights = root / "weights"
result = {
    "root": str(root),
    "image_count": len(images),
    "weight_dirs": sorted(str(p) for p in weights.rglob("inference.yml")),
    "status": "ok" if len(images) == 3352 else "image_count_mismatch",
}
Path("/content/itda_colab_preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
