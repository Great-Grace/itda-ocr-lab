"""Argument-free Colab runner for the current YOLO+PP-OCRv6 validation probe."""
import os
import runpy
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

gpu_install = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "paddlepaddle-gpu==3.3.0", "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu126/"], check=False)
if gpu_install.returncode != 0:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "paddlepaddle==3.3.1"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "paddleocr==3.7.0", "ultralytics==8.4.150", "opencv-python-headless", "psutil", "pyyaml"], check=True)
root = Path("/content/itda_lab")
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
with tarfile.open("/content/itda_ocr_code_bundle.tar.gz", "r:gz") as archive:
    archive.extractall(root)
v6_archive = Path("/content/drive/MyDrive/상품사진입니다/weights/ppocrv6_medium_rec.tar.gz")
v6_root = Path("/content/PP-OCRv6_medium_rec")
if not v6_root.exists():
    v6_root.mkdir(parents=True)
    with tarfile.open(v6_archive, "r:gz") as archive:
        archive.extractall("/content")
    candidates = list(Path("/content").glob("**/inference.yml"))
    found = next((path.parent for path in candidates if "PP-OCRv6_medium_rec" in str(path.parent)), None)
    if found and found != v6_root:
        shutil.copytree(found, v6_root, dirs_exist_ok=True)
os.chdir(root)
os.environ["PYTHONPATH"] = str(root / "src")
sys.argv = [
    "scripts/eval_yolo_paddle_pipeline.py",
    "--weights", "/content/expiry_binary_yolov8n_1280_best.pt",
    "--images", "/content/drive/MyDrive/상품사진입니다",
    "--labels", "data/splits/val.csv",
    "--rec-model-dir", str(v6_root),
    "--rec-model-name", "PP-OCRv6_medium_rec",
    "--device", "0",
    "--rec-device", "gpu:0",
    "--batch-size", "32",
    "--expand", "1.0",
    "--classes", "0",
    "--output", "/content/colab_final_yolo_v6_val.json",
]
runpy.run_path(str(root / "scripts/eval_yolo_paddle_pipeline.py"), run_name="__main__")
