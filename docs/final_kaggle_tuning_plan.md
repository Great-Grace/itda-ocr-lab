# Final Kaggle-guided tuning plan

현재 최고 진단 구조는 `full-image PP-OCRv6 + YOLO region candidate union`이다. 다음 라운드는 후보 recall을 더 올리되, crop-only 회귀를 피하는 방향으로 제한한다.

## Detector sweep

1. `expiry_binary`: Kaggle `date`와 `due`를 하나의 expiry class로 합침. code는 무시.
2. `expiry_full`: expiry와 full만 유지해 semantic hard negative를 보존.
3. `date_due_full`: date/due/full을 별도 유지하되 code를 제거.
4. 각 변형을 YOLOv8n/YOLO11n, input 1280으로 학습.
5. 960/1280 detector 결과를 union하고 class-aware NMS로 후보를 만든다.

## Crop/recognition sweep

각 detector 후보에 대해 crop 확장비 `1.0/1.25/1.5/2.0`을 비교한다. 확장 crop은 기존 full-image OCR branch를 대체하지 않고 보조 인식에만 사용한다.

## Selector sweep

기존 OCR 후보와 detector 후보를 하나의 후보 집합으로 합친 뒤 다음 feature를 고정한다.

- source prior (full OCR / detector crop)
- recognition confidence
- detection confidence
- detector class (`date`, `due`, `full`)
- baseline token bbox와 detector bbox의 IoU
- expiry keyword와의 거리/동일 line 여부
- calendar validity와 pattern reliability

val에서 selector를 고정한 뒤에는 test를 한 번만 실행한다.

## 채택 기준

- Candidate Recall이 기준선보다 3%p 이상 상승
- Selection Accuracy가 90% 이상 유지
- Final EM이 기준선보다 2%p 이상 상승
- CPU 4-core에서 sec/image와 peak RAM이 제출 제약을 만족

문자 집합 제한이나 custom date-only recognizer는 detector union이 개선된 뒤에만 보조 실험으로 재개한다. 지금까지는 해당 방법만으로 유의미한 개선 증거가 없었다.
