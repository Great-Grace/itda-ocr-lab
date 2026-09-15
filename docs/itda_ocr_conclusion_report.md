# 소비기한 OCR: 3,063장 데이터와 CPU 제약에서 확인한 것

## Executive Summary

현재 가장 강한 결론은 단순하다.

> 이 문제의 주된 병목은 날짜를 고르는 규칙이 아니라, 날짜 후보가 만들어지기 전의 인식 과정이다.

459장 `gemini_filtered` validation split에서 측정한 결과:

| 구성 | Final EM | Candidate Recall | Selection Acc. | 속도 | Peak RAM |
|---|---:|---:|---:|---:|---:|
| PP-OCRv5 mobile GPU | 55.74% | 57.05% | 97.70% | 0.248 GPU sec/img | 1.86GB |
| PP-OCRv6 medium GPU | 70.16% | 75.74% | 92.64% | 0.395 GPU sec/img | 1.84GB |
| RapidOCR ONNX + cascade CPU | 64.92% | 67.87% | 95.65% | 0.977 sec/img | 328MB |
| RapidOCR + 저비용 CLAHE CPU | 65.25% | 68.20% | 95.67% | 0.942 sec/img | 303MB |

Selection은 후보가 존재할 때 대체로 93~98%를 맞춘다. 따라서 80% EM을 목표로 하면 selector를 조금 고치는 것보다 Candidate Recall을 약 86% 수준까지 올리는 것이 우선이다.

## 1. 실험 범위와 증거 수준

- 원본 이미지: 3,352장
- 기준 라벨: Gemini-filtered 3,063장
- 물리적 분할: train 2,145 / val 459 / test 459
- 200장 임시 라벨셋: 최종 연구 근거에서 제외
- 공식 leaderboard: CPU fresh run이 완주되기 전까지 등재하지 않음
- 외부 API/VLM: 최종 inference에는 사용하지 않음

팀원의 40장·150장 수치는 방향을 찾는 참고자료로 사용했고, 우리 비교표에는 동일한 459장 split 결과만 넣었다.

## 2. 현재 파이프라인

```text
Image
  → PP-OCR text detector
  → polygon → axis-aligned bbox crop
  → Korean recognizer
  → date candidate parser
  → keyword/spatial selector
  → date normalizer
```

현재 기본 crop은 detector polygon을 받은 뒤 axis-aligned bbox로 잘라서 recognizer에 넣는다. perspective crop 구현은 있으나 기본값이 아니며, 전역 rectification은 팀 실험에서 악화율이 높아 기본 경로로 채택하지 않았다.

## 3. 병목 분해

### Selection은 주범이 아니다

PP-OCRv6 GPU 결과:

- Candidate Recall: 75.74%
- Candidate가 존재할 때 Selection Accuracy: 92.64%

RapidOCR도 Selection Accuracy가 95.65%였다. 후보가 만들어진 뒤에는 비교적 안정적으로 소비기한을 고른다.

### Recognition이 시간과 정확도를 동시에 압박한다

우리 GPU v6 측정에서 평균 stage latency는 detection 약 53ms, recognition 약 306ms였다. 팀원의 CPU 100장 분석에서도 recognition이 전체 처리시간의 약 64%였다.

### Box는 완전히 무너지는 형태가 아니다

VESSL을 재개하지 않고 Google Drive 원본 80장과 기존 PP-OCRv6 token cache를 결합해 overlay를 직접 확인했다. 대부분의 날짜 스탬프 주변에는 detector box가 존재했다. 기존 오류 taxonomy도 DET 0, GEN 16, REC 7, SEL 1로 나타났다.

이는 detector가 전혀 필요 없다는 뜻은 아니다. 다만 polygon 전수 라벨링을 먼저 시작할 만큼 detector miss가 지배적이라는 증거는 아직 없다.

## 4. 팀원·Zotero 연구와의 교차검증

### 팀원 artifact에서 재현된 사실

- 4코어가 8~12코어보다 빠른 CPU 최적점
- box priority로 recognition 비용을 줄일 수 있지만 무조건적인 box cap은 recall을 잃음
- 전역 회전 보정은 악화율이 높음
- RapidOCR ONNX는 Paddle보다 훨씬 빠르지만 정확도는 낮음
- NONE 이미지 중 많은 수가 사람이 읽을 수 있는 도트매트릭스 숫자
- 실패 이미지에만 CLAHE/고해상도 재시도를 붙이는 cascade가 합리적

### Zotero에서 확인한 직접적인 방향

- **PP-OCRv3 / DBNet 계열**: 강한 teacher detector가 만든 pseudo label로 MobileNet 계열 student를 distillation하는 구조
- **DBNet**: differentiable binarization과 hard negative mining으로 text region detector를 강화
- **ASTER**: TPS rectification을 annotation 없이 학습할 수 있으나, 우리 데이터에서는 전역 적용 대신 조건부 적용이 안전
- **SVTRv2**: CTC를 유지하면서 multi-size resize, feature rearrangement, 학습 시 semantic guidance를 사용하고 inference에서는 guidance를 제거
- **CRNN**: character-level 위치 라벨 없이 sequence label만으로 학습 가능하며, synthetic augmentation으로 데이터 부족을 보완
- **STR benchmark 연구**: 모델 수치 비교는 dataset·split·학습 방식이 다르면 쉽게 왜곡되므로 동일 split fresh run이 필수

## 5. Fine-tuning에서 배운 negative result

공식 한국어 PP-OCRv5 pretrained `.pdparams`를 사용한 진짜 PaddleOCR fine-tuning도 수행했다. [PaddleOCR 공식 학습 문서](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md)

| 실험 | Crop Val | End-to-end EM | 판정 |
|---|---:|---:|---|
| pretrained → 30 epoch, lr 5e-4 | 73.91% / 345 crops | 51.48% GPU | reject |
| v5 detector-aligned crop → lr 5e-5 | 41.29% / 201 crops | 미연결 | reject |
| 기존 random-init SVTR/CRNN/Attention | Exact 대부분 0% | 개선 없음 | 연구 참고만 |

Crop 점수가 올랐는데 end-to-end가 떨어진 이유:

1. OCR 후보와 GT가 이미 일치한 쉬운 crop만 학습 데이터로 남김
2. 학습 crop을 만든 detector와 실제 inference detector의 geometry가 다름
3. 도트매트릭스·저대비·인식 실패 이미지가 학습에서 빠짐
4. 실제 pipeline의 box 분포가 training crop 분포와 다름

따라서 “3천장으로 학습했는데 왜 안 오르나”가 아니라, 실제로는 전체 3천장을 학습한 것이 아니며, end-to-end에 맞는 hard crop 학습도 아니었다.

## 6. 앞자리 8을 위한 아키텍처 제안

### 1순위: 학습 없는 Detect Guide

polygon 라벨 없이 바로 검증한다.

- v5/v6/Rapid box agreement
- 날짜 regex 포함 여부
- 소비기한 키워드와 bbox 거리
- box aspect ratio
- recognition confidence

상위 box부터 recognition하고, 후보가 없을 때만 다음 box를 읽는다. 목표는 Candidate Recall과 recognition box 수를 동시에 측정하는 것이다.

### 2순위: Teacher-guided date-region detector

```text
PP-OCRv6/v5 + RapidOCR teacher boxes
  → high-confidence pseudo date-region labels
  → Mobile DBNet/YOLOv8n student
  → date-only crop recognizer
```

처음부터 3,000장 polygon을 수작업으로 만들지 않는다. 100~300장 hard-case bbox를 검수해 pseudo label 품질을 확인하고, 날짜영역 detector가 실제로 Candidate Recall을 올리는지 본다.

### 3순위: 도트매트릭스 selective second-pass

다음 조건에서만 실행한다.

```text
1차 후보 없음
또는 confidence 낮음
또는 dot-pattern/반사 의심
→ local threshold + morphology + 약한 sharpen
→ 숫자·구분자 중심 recognizer
```

전체 이미지 multi-pass는 사용하지 않는다.

### 4순위: detector-aligned pretrained fine-tuning

- 실제 최종 detector가 낸 crop으로 train/val 생성
- 정답 crop뿐 아니라 실패 crop과 hard negative 포함
- backbone 일부 freeze 또는 낮은 learning rate
- 매 epoch마다 crop CER가 아니라 459장 end-to-end EM으로 gate

## 7. 다음 실험의 중단 기준

- Candidate Recall이 2%p 미만 오르면 폐기
- crop CER만 오르고 end-to-end EM이 baseline보다 낮으면 폐기
- 4-core CPU 비용이 0.5초/장 이상 증가하면서 EM 상승이 1%p 미만이면 폐기
- selector만 바꿔 Selection Accuracy가 이미 95% 근처인데 더 최적화하는 실험은 후순위

## 결론

현재 가장 합리적인 방향은 거대한 recognizer를 새로 만드는 것이 아니다.

> 기존 detector box를 먼저 지능적으로 줄이고, 실패한 날짜 crop에만 도트매트릭스 전용 인식기를 적용하는 selective cascade가 80% EM에 가장 가까운 경로다.

polygon 전수 라벨링은 아직 시작하지 않는다. 먼저 100~300장 hard-case box audit로 detector miss 비율을 확정하고, 그 결과가 충분히 높을 때만 date-region student detector를 학습한다.
