#!/usr/bin/env bash
# ==============================================================================
# ITDA Competition - Official Model Weights Setup Script
# Prepares offline detection, recognition, and YOLO models for CPU evaluation.
# Run once by organizers with internet connection before offline evaluation.
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${ROOT_DIR}/weights"
PADDLE_DIR="${WEIGHTS_DIR}/paddle"
YOLO_DIR="${WEIGHTS_DIR}/final_kaggle"

mkdir -p "${PADDLE_DIR}" "${YOLO_DIR}"

echo "=== [ITDA Model Weights Setup Starting] ==="

# 1. Check/Prepare PP-OCRv5 Detection Model
if [ -d "${PADDLE_DIR}/ppocrv5_mobile_det" ] && [ -f "${PADDLE_DIR}/ppocrv5_mobile_det/inference.pdiparams" ]; then
    echo "[PASS] PP-OCRv5 mobile detection model exists locally."
elif [ -n "${ITDA_WEIGHTS_ROOT:-}" ] && [ -d "${ITDA_WEIGHTS_ROOT}/paddle/ppocrv5_mobile_det" ]; then
    echo "[SYNC] Copying PP-OCRv5 detection model from ITDA_WEIGHTS_ROOT..."
    cp -rn "${ITDA_WEIGHTS_ROOT}/paddle/ppocrv5_mobile_det" "${PADDLE_DIR}/" 2>/dev/null || true
else
    echo "[FETCH] Downloading official PP-OCRv5 mobile detection model..."
    python3 -c "from paddleocr import TextDetection; TextDetection(model_name='PP-OCRv5_mobile_det', device='cpu')" 2>/dev/null || true
    if [ -d "${HOME}/.paddlex/official_models/PP-OCRv5_mobile_det" ]; then
        cp -rn "${HOME}/.paddlex/official_models/PP-OCRv5_mobile_det" "${PADDLE_DIR}/"
    fi
fi

# 2. Check/Prepare PP-OCRv6 Medium Recognition Model
if [ -d "${PADDLE_DIR}/PP-OCRv6_medium_rec" ] && [ -f "${PADDLE_DIR}/PP-OCRv6_medium_rec/inference.pdiparams" ]; then
    echo "[PASS] PP-OCRv6 medium recognition model exists locally."
elif [ -n "${ITDA_WEIGHTS_ROOT:-}" ] && [ -d "${ITDA_WEIGHTS_ROOT}/paddle/PP-OCRv6_medium_rec" ]; then
    echo "[SYNC] Copying PP-OCRv6 recognition model from ITDA_WEIGHTS_ROOT..."
    cp -rn "${ITDA_WEIGHTS_ROOT}/paddle/PP-OCRv6_medium_rec" "${PADDLE_DIR}/" 2>/dev/null || true
elif [ -n "${ITDA_WEIGHTS_ROOT:-}" ] && [ -f "${ITDA_WEIGHTS_ROOT}/ppocrv6_medium_rec.tar.gz" ]; then
    echo "[EXTRACT] Extracting PP-OCRv6 archive..."
    tar -xzf "${ITDA_WEIGHTS_ROOT}/ppocrv6_medium_rec.tar.gz" -C "${PADDLE_DIR}/"
else
    echo "[FETCH] Downloading official PP-OCRv6 medium recognition model..."
    python3 -c "from paddleocr import TextRecognition; TextRecognition(model_name='PP-OCRv6_medium_rec', device='cpu')" 2>/dev/null || true
    if [ -d "${HOME}/.paddlex/official_models/PP-OCRv6_medium_rec" ]; then
        cp -rn "${HOME}/.paddlex/official_models/PP-OCRv6_medium_rec" "${PADDLE_DIR}/"
    fi
fi

# 3. Check/Prepare YOLO Expiry Detection Model
YOLO_FILE="${YOLO_DIR}/expiry_binary_yolov8n_1280_best.pt"
if [ -f "${YOLO_FILE}" ]; then
    echo "[PASS] Expiry binary YOLOv8n model exists locally."
elif [ -n "${ITDA_YOLO_WEIGHTS:-}" ] && [ -f "${ITDA_YOLO_WEIGHTS}" ]; then
    echo "[SYNC] Copying YOLO model from ITDA_YOLO_WEIGHTS..."
    cp -f "${ITDA_YOLO_WEIGHTS}" "${YOLO_FILE}"
else
    echo "[FETCH] Downloading trained YOLOv8n expiry detector checkpoint..."
    YOLO_URL="https://github.com/Great-Grace/itda-ocr-lab/releases/download/v1.0.0/expiry_binary_yolov8n_1280_best.pt"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -L -o "${YOLO_FILE}" "${YOLO_URL}" || true
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "${YOLO_FILE}" "${YOLO_URL}" || true
    fi
fi

echo "=== [Verification Summary] ==="
echo "Paddle Det Model: $([ -d "${PADDLE_DIR}/ppocrv5_mobile_det" ] && echo 'READY' || echo 'MISSING')"
echo "Paddle Rec Model: $([ -d "${PADDLE_DIR}/PP-OCRv6_medium_rec" ] && echo 'READY' || echo 'MISSING')"
echo "YOLO Expiry Model: $([ -f "${YOLO_DIR}/expiry_binary_yolov8n_1280_best.pt" ] && echo 'READY' || echo 'OPTIONAL/FALLBACK')"
echo "Setup complete. Model weights are ready in: ${WEIGHTS_DIR}"
