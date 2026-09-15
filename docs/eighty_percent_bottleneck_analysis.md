# 80% EM 돌파를 위한 병목 분석

기준: 최선 실험 `expiry binary YOLOv8n 1280 + PP-OCRv6 full-image/YOLO-crop candidate union`, val 459장.

## 현재 수치

| 항목 | 장수 | 비율 |
| --- | ---: | ---: |
| 정답 | 358 | 77.996% |
| 정답 candidate 존재 | 390 | 84.967% |
| candidate 존재 후 잘못 선택 | 32 | 6.972% |
| 정답 candidate 없음 | 69 | 15.033% |

80%에는 10장만 더 필요하다. 현재 Candidate Recall을 유지하면 selection error 32장 중 10장을 회복해야 한다. Selection을 그대로 두면 candidate miss 69장 중 약 11장을 회복해야 한다.

## Candidate miss 69장 분해

YOLO detector crop 결과를 다시 분류했다.

| detector-crop 상태 | 장수 | 의미 |
| --- | ---: | --- |
| detector box 없음 | 3 | 순수 detector miss |
| box는 있으나 날짜 문자열 없음 | 30 | recognizer blank/non-date output |
| box는 있으나 다른 날짜만 인식 | 36 | recognizer/crop semantic mismatch |

따라서 69 candidate miss 중 **66장(96%)은 detector가 box를 못 찾은 문제가 아니라 crop recognizer가 정답을 만들지 못한 문제**다.

## Selection error 32장 분해

- baseline full-image 후보를 선택해 틀린 경우: 27장
- YOLO crop 후보를 선택해 틀린 경우: 5장

즉 selector는 이미 후보가 있을 때 91.79%를 맞추지만, 기존 full-image 후보의 높은 점수가 detector crop의 정답 후보를 누르는 경우가 남아 있다.

## 결론: 다음 고가치 실험

1. **box-aligned recognizer 학습 데이터 정제**
   - Kaggle expiry box마다 PP-OCRv6 teacher OCR을 수행한다.
   - teacher 결과가 Gemini GT와 정확히 일치하는 crop만 positive transcript로 채택한다.
   - 제조일/코드/다른 날짜가 들어간 crop은 학습에서 제외하거나 hard negative로 둔다.
   - 지금까지 custom recognizer가 실패한 가장 유력한 이유는 이미지-level 날짜 GT를 모든 crop에 그대로 붙여 crop-label mismatch가 발생했기 때문이다.

2. **pretrained recognizer의 좁은 fine-tuning**
   - random-init CTC/Attention이 아니라 PP-OCRv6/SVTR pretrained recognizer의 head 또는 마지막 block만 low-LR로 fine-tune한다.
   - objective는 full string exact와 CER을 함께 사용한다.

3. **selector calibration**
   - 32 selection error 중 baseline source 27장에 대해 detector-crop candidate의 score uplift 조건을 학습 없이 calibration한다.
   - detector bbox와 baseline token bbox의 overlap, detector confidence, recognition confidence, expiry keyword proximity를 feature로 사용한다.

우선순위는 detector 구조를 더 키우는 것이 아니라 `box → 날짜 문자열` 변환의 정확도를 회복하는 것이다.

## 후속 검증 결과

실제로 다음 두 수정을 train에서 고정 학습하고 val에 적용했다.

1. YOLO crop의 영문 월/한국어 날짜 parser를 보정
2. baseline OCR 후보와 YOLO 후보 bbox의 IoU를 selector feature로 추가

결과는 Candidate Recall 84.97% 유지, Selection Accuracy 94.36%, Final EM **80.17%**였다. 따라서 80% 돌파는 새로운 detector나 문자 제한보다 후보 해석과 공간 기반 선택 개선으로 달성됐다.

같은 selector를 fresh GPU OCR 토큰에 적용한 재현 실험에서도 Candidate Recall 85.19%, Selection Accuracy 94.37%, Final EM **80.39%**가 나왔다. 따라서 80% 돌파는 cache에만 의존한 수치가 아니다.
