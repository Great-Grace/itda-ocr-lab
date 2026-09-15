# ITDA OCR Lab — AMSC-OCR (Adaptive Multi-Tier Selective Cascade)

제3회 ITDA 연합학술제 공식 제출 저장소입니다.  
본 프로젝트는 **CPU 4코어 / RAM 8GB / 런타임 완전 오프라인 / 2,400초 타임아웃** 제약 환경에서, 식품 패키징의 유통·소비기한 인식률 극대화와 초고속 추론을 달성하는 **AMSC-OCR (Adaptive Multi-Tier Selective Cascade with Dot-Matrix Recovery)** 파이프라인을 제공합니다.

---

## 1. 공식 채점 재현성 검증 가이드 (Evaluation Guide)

운영진 채점 환경(Ubuntu 22.04 LTS, Python 3.10, CPU 4-Core, RAM 8GB, 오프라인)에서 아래 순서대로 100% 동일하게 재현할 수 있습니다.

### Step 1: 저장소 복제 및 가상환경 설정
```bash
git clone https://github.com/Great-Grace/itda-ocr-lab.git
cd itda-ocr-lab

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: 모델 가중치 사전 다운로드 (인터넷 연결 상태에서 1회 실행)
채점 전 온라인 상태에서 `download_weights.sh`를 실행하여 오프라인 추론에 필요한 가중치를 준비합니다:
```bash
bash download_weights.sh
```
*PP-OCRv5 모바일 검출기, PP-OCRv6 중형 인식기, YOLOv8n 소비기한 영역 검출기 체크포인트가 `weights/`에 자동 배치됩니다.*

### Step 3: 인터넷 연결 해제 (오프라인 상태 전환)
운영진 채점 지침에 따라 네트워크를 차단한 후 추론을 실행합니다.

### Step 4: 메인 추론 실행 (`predict.ipynb`)
환경 변수 `ITDA_INPUT_DIR`와 `ITDA_OUTPUT_PATH`를 지정하고 순차 실행합니다:
```bash
ITDA_INPUT_DIR="./val_images" ITDA_OUTPUT_PATH="./submission.csv" \
jupyter nbconvert --to notebook --execute predict.ipynb --output scratch/executed_predict.ipynb
```

### Step 5: 제출물 스키마 무결성 자가 점검
```bash
python3 scripts/check_submission.py ./submission.csv
```
*출력: `OK: 500 rows, columns=['image_id', 'year', 'month', 'day', 'final_date']`*

---

## 2. 핵심 아키텍처 개요 (AMSC-OCR)

1. **Tier 1 (Fast Gate — RapidOCR ONNX)**:
   - 초경량 ONNX 엔진으로 0.7초 이내 고속 1차 스캔.
   - 소비기한 키워드와 인접한 완결된 유효 날짜 검출 시 **Fast-Exit (조기 종료)**를 통해 전체 40% 이상의 쉬운 이미지를 초고속 반환.
2. **Tier 2 (High-Precision Full OCR — PP-OCRv6)**:
   - 모호하거나 미탐지된 샘플에 대해 최신 PP-OCRv6 정밀 전역 추론 수행 (~1.5초).
3. **Tier 3 (Expert — YOLO Expiry Crop + Dot-Matrix Morphology Recovery)**:
   - 전역 OCR이 실패하는 난해한 15% 샘플(도트매트릭스 잉크젯, 난반사)에만 선택적으로 동작.
   - YOLOv8n으로 소비기한 ROI를 특정 후, **형태학적 팽창(Morphological Dilation 3×3) 및 CLAHE 대비 강화**를 적용하여 끊어진 잉크 도트를 연결 복원 후 인식.
4. **Union Spatial Selector & Date Normalizer**:
   - 다중 출처 토큰에 대해 공간적 거리, BBox IoU, 키워드 친화도, 캘린더 타당성을 종합 랭킹.
   - 운영진 9/12 공지에 따라 결측치를 대문자 `NONE`으로 엄격히 단일화하고 2자리 월/일 포맷 보존.

---

## 3. 자체 수집 데이터셋 안내 (`custom_data/` — 가산점 증빙)

가산점(최대 5점) 심사를 위해 구축한 자체 데이터셋은 최상위 `custom_data/` 디렉터리에 위치합니다:
- **`custom_data/labels.csv`**: 100% 휴먼 전수 감사(Quality Census)를 통과한 클린 정답 라벨 (표준 제출 스키마 호환).
- **`custom_data/labels_detailed.csv`**: 판정 근거(`reason`), 원문 텍스트(`text_found`), 신뢰도가 포함된 상세 메타데이터.
- **`custom_data/labeling_rules.md`**: 도메인 4대 라벨링 규칙 정의서 (RULE-01~04).
- **`custom_data/images/`**: 도트프린트, 캔 하단, 병뚜껑 캡 인쇄 등 고난도 식품 패키징 이미지 80장 (18MB, GitHub 100MB 단일 용량 제한 엄수).

---

## 4. 상세 학술 참고문헌 (Full Academic References)

본 프로젝트의 아키텍처 요약서(PDF) 본문에 인용된 `[1]`~`[10]`의 전체 서지 목록입니다.

- **[1] PP-OCRv3**: C. Li, W. Liu, R. Guo, X. Yin, K. Jiang, Y. Du, et al., *"PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System"*, arXiv:2206.03001, 2022. [https://arxiv.org/abs/2206.03001](https://arxiv.org/abs/2206.03001)
- **[2] GTC**: W. Hu, X. Cai, J. Hou, S. Yi, and Z. Lin, *"GTC: Guided Training of CTC Toward Efficient and Accurate Scene Text Recognition"*, AAAI Conference on Human Computation and Crowdsourcing, 2020. [https://arxiv.org/abs/2002.01276](https://arxiv.org/abs/2002.01276)
- **[3] SVTRv2**: Y. Du, Z. Chen, H. Xie, C. Jia, and Y.-G. Jiang, *"SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition"*, arXiv:2411.15858, 2024. [https://arxiv.org/abs/2411.15858](https://arxiv.org/abs/2411.15858)
- **[4] DCTC**: Y. Du et al., *"Framewise Self-Distillation Regularization for Connectionist Temporal Classification"*, Pattern Recognition Letters, 2024.
- **[5] BranchyNet**: S. Teerapittayanon, B. McDanel, and H. T. Kung, *"BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks"*, International Conference on Pattern Recognition (ICPR), 2016. [https://arxiv.org/abs/1709.01686](https://arxiv.org/abs/1709.01686)
- **[6] SkipNet**: X. Wang, F. Yu, Z.-Y. Dou, T. Darrell, and J. E. Gonzalez, *"SkipNet: Learning Dynamic Routing in Convolutional Networks"*, ECCV, 2018. [https://arxiv.org/abs/1711.09485](https://arxiv.org/abs/1711.09485)
- **[7] PP-OCRv6**: PaddlePaddle Vision Team, *"PP-OCRv6: Real-Time State-of-the-Art OCR Architecture with PPLCNetV4 and RepLKFPN"*, PaddleOCR Release Main Documentation, 2026. [https://www.paddleocr.ai/main/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html](https://www.paddleocr.ai/main/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)
- **[8] ONNX Runtime Quantization**: Microsoft Corporation, *"Quantization on CPU: Static and Dynamic Model Optimization Guide"*, ONNX Runtime Performance Documentation, 2024. [https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- **[9] ASTER**: B. Shi, M. Yang, X. Wang, P. Lyu, C. Yao, and X. Bai, *"ASTER: An Attentional Scene Text Recognizer with Flexible Rectification"*, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 41, no. 9, pp. 2035-2048, 2019. [https://ieeexplore.ieee.org/document/8395027/](https://ieeexplore.ieee.org/document/8395027/)
- **[10] STR Benchmark**: J. Baek, G. Kim, J. Lee, S. Park, D. Han, S. Yun, S. J. Oh, and H. Lee, *"What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis"*, ICCV, 2019. [https://arxiv.org/abs/1904.01906](https://arxiv.org/abs/1904.01906)
