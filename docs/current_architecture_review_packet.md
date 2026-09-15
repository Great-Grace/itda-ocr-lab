# ITDA OCR Lab — 현재 최종 아키텍처 리뷰 패킷

작성일: 2026-09-14  
목적: 다른 AI/연구자가 현재 구조, 실측 근거, 재현 조건, 한계를 독립적으로 검토할 수 있도록 정리한 문서

## 0. 한 문장 결론

현재 가장 좋은 구조는 **PP-OCRv6 전체 이미지 OCR을 유지하면서, Kaggle 동일 원본의 `date/due` box로 학습한 expiry detector를 보조 후보 생성 branch로 추가하고, 두 후보의 bbox 공간 겹침을 포함한 selector로 최종 날짜를 고르는 하이브리드 구조**다.

ITDA validation 459장 fresh GPU 추론에서:

| Metric | Result |
| --- | ---: |
| Final Exact Match | **80.39%** |
| Candidate Recall | **85.19%** |
| Selection Accuracy \| Candidate Exists | **94.37%** |

이 수치는 대회 내부 validation의 최고 실측치다. 외부 표준 benchmark와 동일 조건으로 비교하지 않았으므로 학술적 “세계 SOTA”라고 주장하지 않는다.

## 1. 데이터와 실험 경계

### ITDA 라벨

- 원본 이미지: 3,352장
- Gemini-filtered 날짜 라벨: 3,063장
- train: 2,145장
- val: 459장
- test: 459장
- train/val/test filename overlap: 0
- 모든 라벨의 `confidence=high`이지만 인간 검증 GT가 아니라 자동 라벨이다.

### Kaggle 주석

Kaggle `kimhyeongminkhu/korean-expiry-date-ocr` v3는 약 2.73GB, 23,386개 파일이며 세 단계 주석을 포함한다.

| Subset | Classes | 사용 방식 |
| --- | --- | --- |
| `expiry_region_detection_dataset` | `date`, `due`, `code`, `full` | 최종 detector 학습 |
| `expiry_date_detection_dataset` | `date`, `due`, `code` | crop 구조 확인 및 보조 실험 |
| `expiry_dmy_detection_dataset` | `Y4`, `ENG_MON`, `NUM2` | 구성요소 detector 후보, 최종 구조에는 미사용 |

`region`의 train 2,145장과 val 459장은 우리 split 이미지와 CRC32로 전부 매칭됐다. 별도 box-audit 80장도 전부 매칭됐고, 003181은 SHA-256까지 동일했다. 따라서 이번 결과의 핵심은 추가 이미지 개수가 아니라 **동일 원본에 대한 위치(box) 라벨**이다.

## 2. 최종 아키텍처 상세

```text
원본 상품 이미지
        ├─ Branch A: PP-OCRv5 mobile detector
        │           → PP-OCRv6 medium recognizer
        │           → 전체 OCR token/date 후보
        │
        └─ Branch B: YOLOv8n expiry binary detector
                    (Kaggle date + due → expiry class)
                    → box crop (expand=1.0, margin 6px)
                    → PP-OCRv6 medium recognizer
                    → detector 후보

두 후보 집합 union
        → 날짜 parser/달력 검증
        → confidence + detector score + bbox IoU spatial selector
        → YYYY-MM-DD 출력
```

### Branch A: 전체 이미지 OCR

- Detector: `PP-OCRv5_mobile_det`
- Recognizer: `PP-OCRv6_medium_rec`
- 원본 이미지를 그대로 입력한다.
- detector 설정의 대표값: `det_thresh=0.25`, `box_thresh=0.60`, `unclip_ratio=1.8`, max side 960, recognition batch 8.
- 한국어 keyword(`소비기한`, `유통기한`, `품질유지기한`, `EXP`, `BEST BEFORE`, `USE BY`, `까지`)와 제조일/LOT 음성어를 token context로 보존한다.
- 날짜 후보 생성은 OCR 전체 문자열에서 여러 형식을 만들고, calendar validity를 검사한다.

### Branch B: Kaggle box detector

- Model: `YOLOv8n`
- Input: 1280px
- 학습 class: Kaggle `date`와 `due`를 하나의 `expiry` class로 통합. `code`는 무시.
- Train/val: ITDA train 2,145 / val 459에 대응하는 Kaggle box만 사용.
- Best detector: mAP50 0.9572, mAP50-95 0.6701, weight 약 6MB.
- 추론 시 detector box를 작은 margin만 포함해 320×48 crop으로 만들고 PP-OCRv6에 전달한다.
- box crop branch만 단독으로 쓰지 않는다. 좁은 crop은 제조일·코드·다른 날짜 문맥을 잃을 수 있기 때문이다.

### Union selector

각 후보에는 다음 정보가 남는다.

- normalized date candidate
- source: `baseline` 또는 `yolo`
- recognition confidence
- detection confidence
- baseline OCR candidate score
- detector/token bbox
- baseline 후보 bbox와 detector bbox의 IoU

train 2,145장으로 selector weight를 고정하고 val 459장에 적용했다. 현재 weight는 다음과 같다.

```json
{
  "yolo_bonus": 1.5,
  "candidate_score": 1.5,
  "rec_log": 1.5,
  "det_log": 0.5,
  "spatial_iou": 0.5
}
```

최종 parser는 숫자 날짜뿐 아니라 영문 월(`MAY`, `OCTOBER` 등), 한국어 `년/월/일`, day-first 형식을 지원한다. 이전 임시 parser가 `2021 MAY 26`을 잘못 해석하던 버그를 수정한 뒤 EM이 상승했다.

## 3. 단계별 성능 변화

### Fresh OCR 비교

| Configuration | Split | Final EM | Candidate Recall | Selection | sec/img | Peak RAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PP-OCRv5 mobile + Korean mobile | screen32 | 34.38% | 34.48% | 100.00% | 9.76 | 1.91GB |
| PP-OCRv6 medium | dev100 | 69.00% | 75.27% | 98.57% | 1.83 | 1.38GB |
| PP-OCRv6 medium, box threshold 0.60 | lock100 | 70.00% | 76.34% | 97.18% | 1.49 | 1.39GB |
| PP-OCRv5 server recognizer | dev100 | 64.00% | 69.89% | 98.46% | 2.42 | 1.28GB |
| PP-OCRv6 perspective crop | lock100 | 65.00% | 69.89% | 98.46% | 1.83 | 1.40GB |
| PP-OCRv6 grayscale/contrast | screen32 | 56.25% | 70.00% | 85.71% | 1.61 | 1.32GB |

전체 fresh 실측 43개 행은 [benchmark_all.csv](../runs/benchmark_all.csv)에 있다. 화면 32장, dev100, lock100은 모집단이 다르므로 숫자를 직접 섞어 순위를 만들면 안 된다.

### Kaggle detector/union 비교

| Detector/조합 | mAP50 | Candidate Recall | Final EM |
| --- | ---: | ---: | ---: |
| Region YOLOv8n + v6 crop + 기존 OCR union | 0.9039 | 84.75% | 77.56% |
| Expiry binary YOLOv8n + v6 crop | 0.9572 | 72.33% | 67.76% |
| Expiry binary + parser 보정 + fixed union | 0.9572 | 84.97% | 78.43% |
| Expiry binary + train-fit selector | 0.9572 | 84.97% | 78.87% |
| **Expiry binary + train-fit spatial selector** | 0.9572 | **85.19%** | **80.39%** |
| Expiry/full YOLO11n + union | 0.9607 | 84.75% | 77.12% |
| Date/due/full YOLO11n + union | 0.9459 | 84.10% | 76.69% |

Detector mAP 1위는 `expiry/full YOLO11n`이지만, 최종 날짜 EM 1위는 `expiry binary YOLOv8n + spatial selector`다. 이 문제에서는 mAP보다 OCR 후보 recall과 선택 안정성이 중요하다.

### Fresh 재현 결과

같은 구조를 fresh GPU OCR token과 fresh YOLO 후보에 다시 적용한 결과:

| Metric | Fresh result |
| --- | ---: |
| Full-image PP-OCRv6 baseline EM | 69.28% |
| Full-image Candidate Recall | 75.38% |
| Union Candidate Recall | **85.19%** |
| Selection Accuracy \| Candidate Exists | **94.37%** |
| Union Final EM | **80.39%** |

따라서 80%는 이전 cache에만 존재하는 수치가 아니다.

## 4. 병목과 오류 분해

최선 cached union diagnostic에서 459장을 분해하면:

| 상태 | 장수 |
| --- | ---: |
| 정답 | 358 |
| 정답 candidate 존재 | 390 |
| candidate 존재 후 selection error | 32 |
| 정답 candidate 없음 | 69 |

candidate miss 69장을 detector 결과와 대조하면:

| 상태 | 장수 |
| --- | ---: |
| detector box 없음 | 3 |
| box는 있으나 날짜 문자열 없음 | 30 |
| box는 있으나 다른 날짜만 인식 | 36 |

즉 candidate miss의 66/69(96%)는 detector가 아니라 **crop recognizer 또는 crop 의미 불일치**다. Selection error 32장 중 27장은 baseline OCR 후보가 잘못 선택한 경우, 5장은 detector 후보가 잘못 선택한 경우다. bbox IoU feature를 추가하자 selection이 91.79%에서 94.37%로 상승했다.

## 5. 시도했지만 채택하지 않은 것

| 실험 | 결과 | 판단 |
| --- | --- | --- |
| numeric-date sanitizer | dev100 EM 69% → 65% | 숫자 보정이 오인식을 늘려 폐기 |
| global grayscale/contrast | screen32 EM 71.9% → 56.3% | 전체 전처리 금지, 선택적 fallback만 유지 |
| box crop 단독 | union 전 detector branch EM 67~68% | 문맥 손실 때문에 단독 사용 금지 |
| crop expansion 1.25~1.5 | recall 하락 | expand=1.0 유지 |
| random-init SVTR/CRNN/Attention | exact 0% 또는 낮은 CER | pretrained 기반 없이는 실전 불충분 |
| clean crop PP-OCRv5 fine-tuning | clean crop 정확도는 60.99→72.25%, 실제 union EM은 74.51% | crop domain overfit, 최종 제외 |
| PP-OCRv5 server recognizer | v6보다 낮은 EM, 더 느림 | v6 medium 유지 |

## 6. 환경 적합성 감사

| 항목 | 검증 | 상태 |
| --- | --- | --- |
| 학습 GPU | A100 SXM 80GB ×2, CUDA/Paddle 정상 | PASS |
| 최종 대상 inference | CPU-only, 4-core 제약을 기준으로 설계 | PASS |
| 실제 CPU thread | `taskset 0–3`, OMP/MKL/OPENBLAS=4 | PASS |
| 외부 API/VLM | inference에서 사용하지 않음 | PASS |
| 모델 weight | local materialize, inference 중 다운로드 없음 | PASS |
| split leakage | train/val/test filename-disjoint | PASS |
| Kaggle box leakage | 우리 train ID만 detector 학습, val box는 평가용 | PASS |
| CPU detector branch latency | 459장, 0.863초/장 | PASS |
| full-image CPU latency | 동일 4-core historical fresh 4.87초/장 | PASS |
| full-union CPU exact accuracy | 459장 fresh 전체 재실행은 37분 이상이라 미완료 | **OPEN** |

따라서 **학습 제약과 CPU latency는 맞게 검사했지만, 80.39% 정확도 자체는 GPU fresh 결과**다. GPU와 CPU의 Paddle backend 결과가 완전히 동일하다고 가정하지 않으려면 최종 제출 직전에 CPU full-union 1회를 추가하는 것이 안전하다.

### Colab 재개 확인 (T4)

쿼터 복구 후 Colab T4 세션에서 동일한 `expiry_binary_yolov8n_1280_best.pt` detector의 459장 throughput sweep을 수행했다. Colab 기본 이미지는 Python 3.13이라 Paddle GPU wheel이 제공되지 않아 PP-OCRv6 인식 정확도 실험은 Colab에서 재현하지 못했고, detector 계측만 정확히 기록했다.

| 설정 | sec/img | img/s | box가 나온 이미지 | peak GPU RAM |
| --- | ---: | ---: | ---: | ---: |
| 960 / conf .20 | 0.0446 | 22.42 | 454/459 | 1.70 GB |
| 1280 / conf .20 | 0.0559 | 17.90 | **455/459** | 3.00 GB |
| 1536 / conf .20 | 0.0671 | 14.91 | 454/459 | 4.30 GB |

T4 결과는 `960/.20`을 1차 경량 경로, `1280/.20`을 불확실 샘플의 fallback으로 두는 설계를 지지한다. 1536은 box coverage 증가가 없어 제외한다. 이 표의 “box가 나온 이미지”는 candidate recall이 아니므로 최종 EM 표에 직접 합산하지 않는다. 원자료는 [t4_yolo_sweep.json](../runs/colab/t4_yolo_sweep.json), 요약은 [t4_yolo_sweep.md](../runs/colab/t4_yolo_sweep.md)다.

### Colab T4 정확도 교차검증

동일한 validation ID 459개를 명시적으로 고정해 PP-OCRv6까지 포함한 accuracy run도 완료했다.

| 구조 | Final EM | Candidate Recall | Selection (candidate 존재 시) | sec/img |
| --- | ---: | ---: | ---: | ---: |
| YOLO expiry crop 단독 (960) | 67.76% | 72.33% | 93.67% | 0.532 |
| YOLO expiry crop 단독 (1280) | 66.23% | 71.90% | 92.12% | *warm-cache timing* |
| 전체 PP-OCRv5 det + PP-OCRv6 rec | 71.68% | 75.60% | 94.81% | 0.425 |
| **전체 OCR + YOLO union + train-fit spatial selector** | **80.61%** | **85.19%** | **94.63%** | ~0.957* |

`~0.957 sec/img`는 두 branch를 순차 실행한 상한(0.425 + 0.532)이며, 조건부 fallback에서는 이보다 짧아진다. YOLO-only는 전체 OCR보다 EM이 낮으므로 hard replacement가 아니라 후보 보강 branch로만 사용해야 한다. 자세한 표와 원자료는 [t4_accuracy_summary.md](../runs/colab/t4_accuracy_summary.md)와 [t4_full_yolo_union_trainfit_selector.json](../runs/colab/t4_full_yolo_union_trainfit_selector.json)이다.

## 7. SOTA 판정

### 주장 가능한 것

- 현재 저장된 실측 중 ITDA validation 최고 결과: 80.39% EM
- Kaggle 동일 원본 box supervision을 후보 recall 개선으로 연결한 task-specific hybrid 구조
- Candidate Recall 85.19%, Selection 94.37%라는 병목 분해 근거
- 4-core CPU에서 동작 가능한 local-weight 오프라인 설계

### 주장하면 안 되는 것

- 외부 학술 benchmark 기준 세계 SOTA
- 잠금 test 최종 점수
- invalidated 53-combination master leaderboard의 80.56% 수치
- CPU full-union 정확도가 GPU와 동일하다는 확정적 주장

`runs/multi_architecture_leaderboard/master_leaderboard.md`는 단일 token cache와 hard-coded 보정값을 사용한 시뮬레이션으로 감사 무효화되었고, 이 보고서의 성능표에서 제외했다.

## 8. 리뷰어가 확인할 핵심 질문

1. train-fit spatial selector가 다른 split/제품군에서도 94% 수준 selection을 유지하는가?
2. Kaggle `date/due` box가 실제 expiry인지 manufacturing인지 구분하는 별도 보조 신호가 필요한가?
3. PP-OCRv6 crop recognizer의 66/69 실패를 줄일 수 있는 pretrained fine-tuning 또는 dot-matrix branch가 있는가?
4. CPU full-union 5.7초/장(약 3,352장에 5시간 이상)이 대회 운영상 허용되는가?
5. 최종 test 1회에서 80%대가 재현되는가?

## 9. 재현 자료

- [HTML 전체 보고서](../runs/final_sota_report.html)
- [최종 sweep 보고서](../runs/vessl/final_sweep/final_sweep_report.md)
- [공간 selector weight](../configs/selector/expiry_binary_union_spatial_v2.json)
- [fresh selector fit 결과](../runs/vessl/cpu_final_val/fresh_spatial_selector_fit.json)
- [Kaggle 데이터 검증](kaggle_dataset_utilization.md)
- [80% 병목 분석](eighty_percent_bottleneck_analysis.md)
- [전체 저장 benchmark inventory](../runs/benchmark_all.csv)
