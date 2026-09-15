# 효과 attribution 보고서

검증 기준: ITDA val 459장. 모든 수치는 내부 Gemini-filtered 라벨 기준이며, test에는 아직 적용하지 않았다.

## 결론

가장 큰 효과는 **Kaggle의 동일 원본 이미지에 대한 region bounding-box 라벨을 이용해 detector 후보 branch를 추가한 것**에서 나왔다. 문자열/문자 집합을 제한한 것 자체는 유의미한 개선으로 확인되지 않았다.

## 수치 비교

| 구성 | Candidate Recall | Final EM | 해석 |
| --- | ---: | ---: | --- |
| PP-OCRv6 full-image baseline | 75.16% | 71.24% | 기존 기준선 |
| YOLOv8n box crop + v6, naive | 71.68% | 27.45% | box만 강제하면 악화 |
| YOLOv8n crop + v6, selector 진단 튜닝 | 71.68% | 67.10% | 선택 규칙을 고쳐도 recall 한계 |
| full OCR 후보 + YOLOv8n 후보 union | **84.75%** | **77.56%** | 현재 최고 진단 결과 |
| full OCR 후보 + YOLO11n 후보 union | 84.97% | 77.12% | recall은 높지만 selector 조합은 소폭 열세 |

YOLOv8n detector 자체는 mAP50 0.9039, YOLO11n은 0.9081이었다. 따라서 detector 학습은 실제 box를 잘 배우고 있지만, 검출 box를 단일 crop으로 강제하는 것이 아니라 기존 OCR 후보와 union해야 효과가 난다.

## Kaggle 라벨의 어떤 부분이 효과적이었나

- 80장 SHA-256/전체 train 2,145장·val 459장 CRC 매칭으로, `region` 주석이 우리 동일 원본 이미지임을 확인했다.
- 효과를 낸 것은 추가 이미지 수가 아니라 `date`, `due`, `full` 위치 라벨이다.
- 이 라벨로 detector가 기존 OCR이 놓친 날짜 영역을 별도 후보로 제공했고, Candidate Recall이 75.16% → 84.75%(+9.59%p)로 상승했다.
- `expiry_date` crop 전체를 그대로 recognizer에 넣거나, box 하나만 강제로 crop한 경우에는 v6 crop EM 약 30%, YOLO crop naive EM 27.45%로 오히려 나빠졌다. 날짜/제조일/코드가 섞인 box 의미를 무시했기 때문이다.
- `Y4`, `ENG_MON`, `NUM2` DMY 주석은 아직 최종 recognizer 성능을 직접 올린 증거가 없다. 후속 보조 supervision 후보로만 보류한다.

## 문자열 제한의 효과

현재까지는 **문자 집합 제한이 개선을 만들었다고 볼 근거가 없다.**

- numeric-date sanitizer는 100장 cache 실험에서 EM 69% → 65%, selection 98.57% → 86.67%로 악화됐다.
- date-only CTC/Attention custom recognizer들은 2,145 train / 459 val crop에서 20~30 epoch 후에도 exact match가 0%였다(CER 최선 약 0.53 수준).
- 공식 PP-OCRv5 recognizer fine-tuning도 end-to-end EM 55.74% → 51.48%로 하락했다.
- 이유는 문자열 제한이 이미 잘린/잘못 crop된 이미지를 복구하지 못하기 때문이다. 숫자 allowlist는 OCR 이후 후보 정제에는 도움을 줄 수 있지만, 인식하지 못한 날짜를 새로 만들어내지 못한다.

## 해석

현재 병목은 selector 자체가 아니라 **후보 생성 recall(특히 recognition/detection 경계)**이다. Selection은 후보가 존재할 때 대체로 90% 이상 유지된다. 따라서 다음 주력 구조는:

```text
full-image PP-OCRv6 후보
        +
YOLOv8n region date/due 후보
        ↓
하나의 고정 confidence/spatial/keyword selector
```

이며, detector box로 전체 OCR을 대체하지 않는다. 77.56%는 val에서 selector를 진단 조정한 값이므로, 최종 제출 전 독립 holdout 또는 잠금 test에서 한 번만 확인해야 한다.
