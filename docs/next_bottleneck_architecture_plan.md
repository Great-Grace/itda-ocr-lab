# 병목 해결용 다음 아키텍처 계획

## 목표 수치

현재 PP-OCRv6 GPU 결과는 Final EM 70.16%, Candidate Recall 75.74%,
Selection Accuracy 92.64%다. Selection이 그대로라고 가정하면 80% EM을
만들기 위해 필요한 Candidate Recall은 대략 `0.80 / 0.9264 = 86.4%`다.
따라서 다음 연구의 목표는 selector 점수 조정이 아니라 Candidate Recall을
약 10%p 끌어올리는 것이다.

## Zotero에서 확인한 직접 근거

### 1. Teacher-guided date-region detector

PP-OCRv3 논문은 DB teacher를 먼저 강화한 뒤 CML로 MobileNet 계열 student를
distillation하는 구조를 쓴다. 우리 문제에서는 강한 PP-OCRv6/v5 detector와
RapidOCR의 box를 teacher ensemble로 사용하고, 날짜 가능성이 높은 영역만
student가 학습하도록 한다.

우리 변형:

`PP-OCRv6/v5 + RapidOCR boxes → confidence/GT-date consistency filter →
date-region DBNet-Mobile/YOLOv8n student → date crop recognizer`

핵심은 일반 text box가 아니라 `expiry-date region`을 학습하는 것이다. GT 날짜와
일치하는 teacher box는 positive, 제조일/LOT/바코드/영양성분 box는 hard negative로
넣는다. 400~600장 정도의 수동 polygon 보정이 가능하면 pseudo-label 품질을
검증하는 작은 anchor set으로 쓴다.

### 2. Label-free detect guide / ROI ranking

polygon 라벨을 바로 만들기 어렵다면 먼저 학습 없이 구현한다.

- OCR box의 aspect ratio, height, text confidence
- `소비기한/유통기한/EXP/까지`와의 거리
- 날짜 regex 후보와의 겹침
- v5/v6/Rapid 세 엔진의 box agreement

로 box를 정렬하고, 상위 box부터 recognition한다. 날짜 후보가 생기면 뒤의 box는
인식하지 않는다. 팀 artifact의 box priority 실험은 recognition 비용을 51% 줄였지만
정답 손실 사례가 있었으므로, `No candidate → 다음 box` 안전 게이트를 반드시 둔다.

### 3. 도트매트릭스 전용 second-pass

Zotero의 ARABEX/도트 OCR 연구는 점 패턴을 획처럼 정규화한 뒤 CRNN을 학습한다.
우리 데이터에서는 전체 이미지가 아니라 다음 조건에서만 실행한다.

`1차 candidate 없음 OR 인식 confidence 낮음 OR dot-pattern score 높음`

전처리는 adaptive threshold, morphology close, local contrast, 약한 perspective
warp를 조합하고, recognizer는 한국어 전체가 아니라 숫자·구분자 중심의 날짜 crop
recognizer로 제한한다. 모델은 synthetic dot-font date와 실제 hard crop을 섞어
학습하되, validation은 실제 사진만 사용한다.

### 4. SVTRv2식 recognizer 학습

SVTRv2 논문에서 가져올 것은 Attention decoder 자체가 아니라 다음 세 가지다.

- aspect ratio에 따른 multi-size resize
- 2D feature rearrangement로 비정렬/기울어진 글자 대응
- 학습 시 semantic guidance를 사용하고 inference에서는 제거

현재 official PP-OCRv5 구조가 이미 CTC/NRTR multi-head와 multi-scale sampler를
포함하므로, 먼저 새 모델을 만들기보다 `pretrained PP-OCRv5 fine-tune +
detector-aligned hard crop`을 검증한다. crop 정확도만 올리고 end-to-end가 떨어진
기존 실험처럼 되지 않도록 매 epoch마다 459장 pipeline EM을 확인한다.

### 5. 조건부 rectification

ASTER의 TPS rectification은 annotation 없이 학습할 수 있지만, 팀 실험에서
무조건적인 방향 보정은 87% 악화됐다. 따라서 전역 보정은 금지하고, polygon의
기울기/원근 왜곡이 큰 crop에만 perspective warp를 적용한다.

## 실험 순서

1. **Box agreement ranking**: 학습 없이 v5/v6/Rapid box agreement + keyword ROI,
   Candidate Recall/recognition box 수/latency 측정.
2. **Conditional perspective crop**: low-confidence/기울어진 crop만 perspective
   warp, 459장 fresh OCR.
3. **Teacher pseudo-label detector**: 400~600장 anchor polygon을 사람이 검수하고
   날짜영역 student detector 학습.
4. **Dot-matrix second-pass**: hard crop + synthetic dot-date augmentation,
   pretrained recognizer의 일부 layer만 fine-tune.
5. **End-to-end gate**: 각 단계는 crop CER가 아니라 `Candidate Recall → Final EM`
   순서로 승격 여부를 결정한다.

## 중단 기준

- Candidate Recall이 2%p 미만 상승하면 selector/recognizer 조합을 폐기
- crop val만 상승하고 459장 EM이 baseline보다 낮으면 즉시 폐기
- 4-core CPU 기준 0.5초/장 이상 추가되면서 EM 상승이 1%p 미만이면 폐기
- test split은 최종 후보 확정 전까지 사용하지 않음
