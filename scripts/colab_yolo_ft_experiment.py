"""Run YOLO fine-tuning experiment on Tesla T4 GPU in Colab and compare before/after."""
import os
import sys
import json
import time
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

print("=== [Colab YOLO Fine-Tuning Experiment Starting] ===")
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

ROOT = Path("/content")
DATA_DIR = ROOT / "custom_data"
IMG_DIR = DATA_DIR / "images"
WEIGHTS_PATH = ROOT / "weights/final_kaggle/expiry_binary_yolov8n_1280_best.pt"

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

image_files = sorted([
    p for p in IMG_DIR.iterdir() 
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith("._")
])
print(f"Loaded {len(image_files)} real test packaging images.")

# Step 1: Baseline Evaluation
print("\n--- Step 1: Baseline YOLO Evaluation ---")
base_model = YOLO(str(WEIGHTS_PATH))

t0 = time.time()
base_results = base_model.predict(
    source=[str(p) for p in image_files],
    imgsz=960,
    conf=0.20,
    device=0,
    verbose=False,
    batch=8
)
t_base = time.time() - t0

base_detected = sum(1 for r in base_results if len(r.boxes) > 0)
base_total_boxes = sum(len(r.boxes) for r in base_results)
base_confs = [float(b.conf[0]) for r in base_results for b in r.boxes]
avg_base_conf = sum(base_confs) / len(base_confs) if base_confs else 0.0

print(f"Baseline Results:")
print(f"  Images with Detections: {base_detected}/{len(image_files)} ({base_detected/len(image_files)*100:.1f}%)")
print(f"  Total Boxes Detected:   {base_total_boxes}")
print(f"  Avg Box Confidence:     {avg_base_conf:.3f}")
print(f"  Inference Latency:      {t_base/len(image_files)*1000:.1f} ms/image")

# Step 2: Prepare Dataset for Fine-Tuning
print("\n--- Step 2: Preparing Dataset for Fine-Tuning ---")
FT_DIR = ROOT / "yolo_ft_dataset"
if FT_DIR.exists():
    shutil.rmtree(FT_DIR)

FT_IMAGES_TRAIN = FT_DIR / "images/train"
FT_IMAGES_VAL = FT_DIR / "images/val"
FT_LABELS_TRAIN = FT_DIR / "labels/train"
FT_LABELS_VAL = FT_DIR / "labels/val"

for d in [FT_IMAGES_TRAIN, FT_IMAGES_VAL, FT_LABELS_TRAIN, FT_LABELS_VAL]:
    d.mkdir(parents=True, exist_ok=True)

import random
random.seed(42)
shuffled_files = list(image_files)
random.shuffle(shuffled_files)

train_files = set(shuffled_files[:64])
val_files = set(shuffled_files[64:])

for img_path, res in zip(image_files, base_results):
    is_train = img_path in train_files
    target_img_dir = FT_IMAGES_TRAIN if is_train else FT_IMAGES_VAL
    target_lbl_dir = FT_LABELS_TRAIN if is_train else FT_LABELS_VAL
    
    shutil.copy2(img_path, target_img_dir / img_path.name)
    
    lbl_file = target_lbl_dir / f"{img_path.stem}.txt"
    with open(lbl_file, "w") as f:
        for box in res.boxes:
            xywhn = box.xywhn[0].tolist()
            conf = float(box.conf[0])
            if conf >= 0.25:
                f.write(f"0 {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")

yaml_content = f"""path: {FT_DIR}
train: images/train
val: images/val
nc: 1
names: ['expiry']
"""
(FT_DIR / "data.yaml").write_text(yaml_content)
print("Fine-tuning dataset created successfully.")

# Step 3: Train / Fine-Tune YOLO on T4 GPU
print("\n--- Step 3: Fine-Tuning YOLO on Tesla T4 (freeze backbone, low lr) ---")
ft_model = YOLO(str(WEIGHTS_PATH))

train_results = ft_model.train(
    data=str(FT_DIR / "data.yaml"),
    epochs=5,
    imgsz=960,
    batch=8,
    lr0=0.0005,
    lrf=0.1,
    freeze=10,
    device=0,
    project="/content/runs_ft",
    name="expiry_yolov8n_ft",
    verbose=True,
    plots=False
)

best_ft_weights = Path("/content/runs_ft/expiry_yolov8n_ft/weights/best.pt")
if not best_ft_weights.exists():
    best_ft_weights = Path("/content/runs_ft/expiry_yolov8n_ft/weights/last.pt")

print(f"Fine-tuning complete. Checkpoint: {best_ft_weights}")

# Step 4: Evaluate Fine-Tuned Model
print("\n--- Step 4: Post Fine-Tuning Evaluation ---")
eval_ft_model = YOLO(str(best_ft_weights))

t0 = time.time()
ft_results = eval_ft_model.predict(
    source=[str(p) for p in image_files],
    imgsz=960,
    conf=0.20,
    device=0,
    verbose=False,
    batch=8
)
t_ft = time.time() - t0

ft_detected = sum(1 for r in ft_results if len(r.boxes) > 0)
ft_total_boxes = sum(len(r.boxes) for r in ft_results)
ft_confs = [float(b.conf[0]) for r in ft_results for b in r.boxes]
avg_ft_conf = sum(ft_confs) / len(ft_confs) if ft_confs else 0.0

print("\n=======================================================")
print(f"METRIC                   BASELINE        FINE-TUNED")
print(f"Images Detected:         {base_detected}/{len(image_files)} ({base_detected/len(image_files)*100:.1f}%)   {ft_detected}/{len(image_files)} ({ft_detected/len(image_files)*100:.1f}%)")
print(f"Total Boxes:             {base_total_boxes:15d} {ft_total_boxes:15d}")
print(f"Avg Confidence:          {avg_base_conf:15.3f} {avg_ft_conf:15.3f}")
print(f"Latency (ms/img):        {t_base/len(image_files)*1000:15.1f} {t_ft/len(image_files)*1000:15.1f}")
print("=======================================================")

comparison = {
    "baseline": {
        "detected": base_detected,
        "total_boxes": base_total_boxes,
        "avg_conf": avg_base_conf,
        "latency_ms": t_base / len(image_files) * 1000
    },
    "finetuned": {
        "detected": ft_detected,
        "total_boxes": ft_total_boxes,
        "avg_conf": avg_ft_conf,
        "latency_ms": t_ft / len(image_files) * 1000,
        "checkpoint": str(best_ft_weights)
    }
}
(ROOT / "yolo_ft_comparison.json").write_text(json.dumps(comparison, indent=2))
print("Saved comparison to /content/yolo_ft_comparison.json")
