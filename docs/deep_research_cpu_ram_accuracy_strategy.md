# CPU-only OCR에서 RAM·학습 GPU·시간 제한을 동시에 쓰는 전략

## Executive conclusion

현재 제약은 500장, 4코어 CPU, RAM 8GB, 2,400초 timeout이다. 따라서 평균 예산은 4.8초/장이고, 제출 안정성을 위해서는 3.6초/장 이하를 목표로 잡는 것이 안전하다.

결론은 “가장 큰 모델을 네 개의 프로세스로 병렬 실행”이 아니다. 4코어에서는 모델 복제와 스레드 경쟁 때문에 오히려 느려질 가능성이 높다. RAM은 다음 세 가지에 써야 한다.

1. 경량 CPU gate, 주력 recognizer, 어려운 crop 전용 expert를 한 번만 메모리에 상주시킨다.
2. 쉬운 이미지에는 빠른 경로만 통과시키고, 후보가 없거나 불확실한 이미지에만 무거운 경로를 실행한다.
3. 학습 GPU에서 teacher·hard-negative·distillation을 활용해 배포 student의 후보 recall을 올린다.

권장 최종 형태는 다음과 같다.

```text
RapidOCR/경량 ONNX gate
  ├─ 날짜 후보·키워드·confidence가 충분함 → 즉시 종료
  └─ 후보 없음·불확실·도트매트릭스 의심
       ↓
     PP-OCRv6 medium full-image
       ├─ 안정적인 후보 → 종료
       └─ 여전히 불확실
            ↓
          expiry detector + 날짜 crop expert
            ↓
          공간·confidence selector
```

현재 full OCR + YOLO union을 모든 이미지에 실행하는 구조는 정확도 후보로는 강하지만, 기록된 5.7초/장이라면 500장에 약 47.5분이다. 반면 주력 PP-OCRv6 CPU baseline은 1.5~1.9초/장으로 약 12~15분이다. 따라서 union을 버리기보다 **조건부 expert**로 바꾸는 것이 RAM과 학습 GPU를 동시에 활용하는 가장 현실적인 방법이다.

## 1. 제약과 예산

### 1.1 시간

운영진 공지 기준 500장 전체에 2,400초가 주어진다.

```text
2,400초 / 500장 = 4.8초/장
```

실험 기록상:

| 구성 | 기록된 속도 | 500장 환산 | 판정 |
|---|---:|---:|---|
| PP-OCRv6 medium CPU | 1.49~1.83초/장 | 12.4~15.3분 | 안전 |
| RapidOCR ONNX cascade | 0.94~0.98초/장 | 7.8~8.2분 | 매우 안전, 정확도 부족 |
| PP-OCRv5 mobile 일부 구성 | 약 9.8초/장 | 약 81분 | 부적합 |
| full OCR + YOLO union 추정 | 약 5.7초/장 | 약 47.5분 | 속도점수 위험 |

목표는 timeout에 딱 맞추는 4.8초가 아니라 **3.6초/장 이하**다. 운영진 CPU 세대, import·모델 초기화, 이미지 분포, 시스템 부하를 위한 여유가 필요하다.

### 1.2 RAM

현재 실측 peak RSS는 PP-OCRv6 medium이 약 1.3~1.4GB, 일부 무거운 Paddle 구성도 2.6GB 수준이다. 8GB는 “큰 batch를 무조건 키우는 자원”이라기보다, **여러 전문가 모델을 상주시킬 수 있는 자원**이다.

다만 RAM이 늘어도 CPU FLOPs가 줄지는 않는다. 큰 모델을 항상 실행하면 메모리 여유가 정확도나 속도로 자동 변환되지 않는다.

## 2. 현재 파이프라인에서 실제 병목

내부 결론 보고서의 병목 분해는 다음과 같다.

- 후보가 존재한 뒤의 selector 정확도는 대체로 93~98%다.
- 최종 정확도를 제한하는 것은 후보가 만들어지기 전의 detection·recognition이다.
- union의 candidate miss 69장 중 66장은 detector box 자체보다 crop recognizer가 날짜 문자열을 만들지 못한 문제였다.
- CPU 시간의 큰 비율은 recognition이 차지한다.

따라서 selector weight를 더 복잡하게 만드는 것보다, **recognition 실패 샘플을 싸게 찾아내고 적절한 expert에 라우팅하는 것**이 우선이다.

현재 실제 코드에는 두 가지 비용 포인트가 있다.

1. full-image Paddle detector/recognizer를 실행한다.
2. YOLO crop branch에서 PP-OCRv6 recognizer 인스턴스를 별도로 다시 로드한다.

두 recognizer가 같은 가중치를 사용하더라도 런타임 객체와 메모리가 자동으로 공유된다고 보장할 수 없다. 이는 peak RAM보다는 모델 초기화와 inference 호출 비용에 더 큰 영향을 줄 수 있다.

## 3. 문헌·공식 문서가 지지하는 방향

### 3.1 CTC·경량 recognizer를 teacher 지식으로 강화

PP-OCRv3는 경량 SVTR-LCNet, CTC guided training, DML/UDML 계열 distillation을 결합해 비슷한 추론 속도에서 정확도를 높이는 방향을 제시했다.[1] GTC도 attention 모델의 지식을 CTC 모델에 전달해 CTC의 빠른 inference를 유지하는 접근이다.[2]

SVTRv2는 CTC 구조가 encoder-decoder보다 빠른 장점을 유지하면서 multi-size resizing, feature rearrangement, semantic guidance를 학습에 사용한다. 특히 semantic guidance는 inference에서 제거할 수 있도록 설계되어, **학습 때는 무겁게, 배포 때는 가볍게**라는 이번 대회의 조건과 잘 맞는다.[3]

우리 팀의 random-init SVTR·CRNN 실패 결과를 고려하면, 새 recognizer를 처음부터 학습하는 것이 아니라 PP-OCRv6/SVTR 계열 pretrained model을 teacher로 사용하고, 실제 detector crop의 hard case를 포함해 마지막 block/head를 low-LR fine-tuning하는 쪽이 타당하다.

### 3.2 self-distillation은 inference 비용을 늘리지 않는 후보

DCTC는 CTC loss에 framewise self-distillation regularization을 추가해 문자별 alignment 학습을 강화하면서, 추가 inference parameter·추가 inference 단계·추가 데이터 없이 정확도 향상을 노린다.[4] 이 방식은 제출 시 teacher를 함께 탑재하지 않아도 된다는 점이 중요하다.

### 3.3 선택적 실행과 early exit

BranchyNet은 confidence가 높은 샘플을 중간 branch에서 조기 종료하고 어려운 샘플만 깊은 경로로 보내는 구조다.[5] SkipNet은 입력별로 convolution block을 선택적으로 건너뛰어 계산량을 줄인다.[6] 이번 문제에 그대로 신경망 early-exit을 새로 구현할 필요는 없지만, **OCR backend 수준의 외부 cascade**로 같은 원리를 구현할 수 있다.

우리 문제의 gate는 다음처럼 만들 수 있다.

- 날짜 정규식 후보가 완전한 형태로 존재하는가
- 소비기한 keyword와 후보 bbox의 거리가 가까운가
- recognition confidence가 threshold 이상인가
- 후보가 둘 이상 충돌하는가
- 점·반사·저대비·세로 비율 등 hard-case 특징이 있는가

### 3.4 runtime이 모델 크기보다 더 큰 레버리지

공식 PaddleOCR 문서는 PP-OCRv6 medium recognizer의 ONNX Runtime CPU end-to-end 수치를 Paddle static보다 낮게 보고하고, OpenVINO CPU 파이프라인에서도 Paddle 대비 큰 차이를 보고한다.[7] 모델과 CPU가 다르므로 우리 서버에 그대로 복사할 수는 없지만, **Paddle Python API만 고집하지 말고 ONNX/OpenVINO backend를 동일 가중치로 계측해야 한다**는 강한 신호다.

ONNX Runtime 공식 문서는 CNN에는 static quantization, RNN/Transformer에는 dynamic quantization을 우선 검토하라고 설명한다. CPU에서 int8 QDQ 경로를 지원하고, calibration data를 사용한 static quantization과 QAT를 구분한다.[8]

이번 recognizer는 CNN + CTC/sequence 요소가 섞여 있으므로 한 번에 전체 int8화하지 말고 다음 순서가 안전하다.

1. FP32 ONNX graph optimization
2. backbone convolution static INT8
3. sequence/attention 부분은 FP32 또는 dynamic INT8
4. 459장 holdout에서 CER·연/월/일 score·latency 동시 비교

### 3.5 정류(rectification)는 전체 적용하지 말고 조건부로

ASTER는 Thin-Plate Spline rectification으로 왜곡 텍스트를 보정하는 대표 연구다.[9] 그러나 우리 실험에서는 전역 grayscale·perspective 보정이 악화된 적이 있다. ASTER류 정류를 전체 이미지에 켜기보다, 경사·세로비·bbox polygon 불안정이 감지된 crop에만 적용해야 한다.

### 3.6 모델 비교는 동일 split·동일 runtime으로

STR benchmark 연구는 dataset, split, training recipe, preprocessing 차이가 모델 순위를 크게 왜곡할 수 있음을 지적한다.[10] “다른 팀이 80%를 넘겼다”는 소문을 판단할 때도 exact match인지, 연·월·일 부분점수인지, GPU인지 CPU인지, hidden 500장인지가 확인되지 않으면 직접 비교할 수 없다.

## 4. 후보 아키텍처 비교

### A. Fast gate → v6 → crop expert: 최우선

**구성**

```text
RapidOCR ONNX 전체
  → 빠른 후보가 확실하면 종료
  → 불확실 샘플에 PP-OCRv6 full-image
  → 아직 불확실한 샘플에 YOLO/date-crop expert
```

**장점**

- RapidOCR와 v6를 RAM에 함께 올려도 peak RAM 여유가 크다.
- 모든 이미지에 무거운 branch를 실행하지 않는다.
- 현재 union의 후보 보강 효과를 유지할 수 있다.
- 학습 GPU에서 gate classifier와 crop expert를 학습할 수 있다.

**핵심 위험**

RapidOCR가 잘못된 후보를 높은 confidence로 내놓으면 v6 fallback이 호출되지 않는다. 따라서 “후보 존재 여부” 하나가 아니라 후보의 완전성·keyword·confidence·source disagreement를 함께 사용해야 한다.

### B. v6 full-image + selective heavy crop recognizer: 현실적인 공격형

full-image v6는 유지하고, YOLO crop이 존재하는 모든 이미지가 아니라 다음에만 무거운 recognizer를 실행한다.

- v6에 완전한 날짜 후보가 없음
- 후보가 partial date임
- v6 후보와 detector crop 후보의 source가 충돌함
- dot-matrix/low-contrast heuristic가 켜짐

현재 union이 약 5.7초/장이고 v6가 1.8초/장이라고 가정하면, heavy branch의 추가 비용은 대략 3.9초/장이다. 총 4.2초/장 목표를 잡으면 heavy branch 호출률을 약 60% 이하로 줄여야 한다. 안전한 운영 목표는 30~40%다.

### C. ONNX/OpenVINO v6 + 기존 selector: 매우 먼저 계측

새 모델을 학습하기 전에 동일 PP-OCRv6 가중치를 ONNX Runtime 또는 OpenVINO로 export해 CPU 추론한다. 이 실험은 정확도 변경을 최소화하면서 runtime만 바꾸므로 가장 해석이 쉽다.

필수 검증:

- text detector와 recognizer 각각의 export 가능 여부
- output text/score/bbox가 Paddle path와 동일한지
- 4코어에서 thread 수 1, 2, 4 비교
- FP32와 INT8의 연·월·일 score 및 CER
- warm-up 포함/제외 latency

### D. branch 병렬 실행: 보조 실험

full OCR와 YOLO detection을 동시에 돌리는 2+2 core 분할은 이론상 union latency를 줄일 수 있다. 그러나 Paddle·Ultralytics가 각자 내부 thread pool을 만들기 때문에 다음을 함께 고정해야 한다.

- Paddle intra-op threads
- ONNX Runtime intra/inter-op threads
- Ultralytics torch threads
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
- CPU affinity

각 branch를 별도 process로 띄우는 방식은 모델 복제 때문에 피한다. 먼저 단일 process의 thread pool 제한으로 측정하고, 실제 wall time이 5% 이상 줄지 않으면 폐기한다.

### E. 무거운 server recognizer를 전체 적용: 비추천

우리 실험에서도 server recognizer는 v6보다 느리고 최종 정확도가 낮았다. 공식 PaddleOCR 표도 server 계열이 항상 더 좋은 CPU latency/정확도 trade-off를 보장하지 않는다고 보여준다.[7] 큰 모델 파일을 상주시킨다는 사실만으로 배포 품질이 좋아지지 않는다.

## 5. 학습 GPU를 가장 잘 쓰는 방법

### 5.1 teacher는 inference가 아니라 label·routing에 사용

GPU에서 다음을 offline으로 수행한다.

1. PP-OCRv6, PP-OCRv5, RapidOCR, YOLO detector 결과를 모두 생성한다.
2. 각 이미지에 대해 후보 recall과 source disagreement를 기록한다.
3. 정답 후보가 존재하는지, 어떤 expert가 회복했는지 라벨링한다.
4. 작은 gate model을 학습해 “다음 expert 호출 여부”를 예측한다.

배포 시에는 gate와 student만 사용하고, GPU teacher는 포함하지 않는다.

### 5.2 crop label mismatch를 제거

이미지-level 최종 날짜를 모든 crop에 붙이지 않는다. detector crop과 실제 OCR 문자열이 일치하는 경우만 positive로 사용하고, 제조일·LOT·다른 날짜는 hard negative로 포함한다.

학습 데이터는 다음 네 묶음으로 구성한다.

- exact expiry crop
- partial date crop
- manufacturing date crop
- code/phone/영양성분 숫자 hard negative

loss는 CTC/character loss + teacher soft logits + candidate-level date validity loss의 조합을 검토한다. 최종 gate는 crop CER가 아니라 459장 end-to-end score로 선택한다.

### 5.3 confidence calibration

recognition confidence는 모델마다 calibration이 다르다. raw confidence를 바로 합산하지 말고 train split에서 temperature scaling 또는 isotonic calibration을 학습해 다음 feature로 사용한다.

- calibrated recognition confidence
- detection confidence
- bbox IoU
- expiry keyword distance
- date completeness
- source agreement

## 6. 실험 우선순위

| 순위 | 실험 | 목표 | 폐기 기준 |
|---:|---|---|---|
| 1 | 동일 v6의 ONNX/OpenVINO CPU export | runtime만 개선 | 정확도 동일·속도 개선 없음 |
| 2 | RapidOCR → v6 confidence cascade | 평균 3.6초/장 이하 | candidate recall 2%p 이상 하락 |
| 3 | v6 → YOLO/crop expert selective routing | union 정확도 유지·호출률 40% 이하 | EM/부분 score +1%p 미만인데 +0.5초/장 이상 |
| 4 | v6 recognizer INT8/PTQ/QAT | crop recognition 속도·RAM 개선 | 날짜 component score 급락 |
| 5 | 2+2 branch parallel | union latency 감소 | 단일 4-thread보다 5% 미만 개선 |
| 6 | server recognizer 전체 적용 | 참고용 | 기존 v6보다 느리거나 정확도 하락 |

### 검증 프로토콜

모든 후보는 같은 459장 holdout에서 먼저 비교한다. 운영진 평가셋은 500장이므로, 최종 후보는 500장과 동일한 이미지 수로 다음 명령을 실행한다.

- network off
- Ubuntu 22.04 / Python 3.10
- 4 CPU core 고정
- 모델 초기화 시간을 포함한 cold run 1회
- warm run 1회
- peak RSS
- 평균·p95 latency
- candidate recall
- 연·월·일 component score
- final exact match
- submission.csv 생성 여부

목표는 다음이다.

```text
평균 3.6초/장 이하
500장 30분 이내
baseline 대비 component score 하락 없음
peak RSS 6GB 이하
```

## 7. 최종 권고

현재 단계에서 “더 무거운 단일 모델”은 좋은 베팅이 아니다. 가장 기대값이 높은 조합은 다음이다.

```text
학습 GPU:
  PP-OCRv6/YOLO/Rapid ensemble로 teacher labels 생성
  hard-negative crop과 routing label 생성
  작은 gate·crop expert fine-tuning

제출 CPU:
  RapidOCR 또는 ONNX v6 fast gate
  → v6 full-image
  → 불확실 샘플만 heavy crop expert
  → 기존 spatial selector
```

8GB는 충분히 활용할 수 있지만, 그 사용처는 model parallelism이 아니라 **resident experts + selective execution + calibration cache**다. 4코어에서 극한 병렬화를 먼저 시도하기보다 ONNX/OpenVINO, selective routing, crop-specific distillation 순으로 계측하는 편이 안전하다.

80% 소문리는 아직 판단 근거로 쓰지 않는다. 우리 내부 결과도 GPU validation에서는 이미 80%대였지만, 공식 CPU 500장 hidden score와 직접 비교한 것은 아니다. 먼저 같은 split에서 candidate recall과 component score를 올리고, 그 다음 500장 cold-run timeout을 확인하는 것이 맞다.

## Sources

1. Chenxia Li et al., “PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System,” arXiv:2206.03001. [arXiv](https://arxiv.org/abs/2206.03001). Zotero item: `QLPNWL59`.
2. Wenyang Hu et al., “GTC: Guided Training of CTC Towards Efficient and Accurate Scene Text Recognition,” arXiv:2002.01276. [arXiv](https://arxiv.org/abs/2002.01276).
3. Yongkun Du et al., “SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition,” arXiv:2411.15858. [arXiv](https://arxiv.org/abs/2411.15858). Zotero item: `XEBTJC2I`.
4. Ziyin Zhang et al., “Self-distillation Regularized Connectionist Temporal Classification Loss for Text Recognition,” arXiv:2308.08806. [arXiv](https://arxiv.org/abs/2308.08806).
5. Surat Teerapittayanon et al., “BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks,” arXiv:1709.01686. [arXiv](https://arxiv.org/abs/1709.01686).
6. Xin Wang et al., “SkipNet: Learning Dynamic Routing in Convolutional Networks,” arXiv:1711.09485. [arXiv](https://arxiv.org/abs/1711.09485).
7. PaddleOCR, “Text Recognition Module,” “PP-OCRv6,” and “Inference Engine and Configuration.” [Text Recognition](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html), [PP-OCRv6](https://www.paddleocr.ai/main/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html), [Inference Engine](https://www.paddleocr.ai/main/en/version3.x/inference_deployment/local_inference/inference_engine.html).
8. ONNX Runtime, “Quantize ONNX models.” [Official documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
9. Baoguang Shi et al., “ASTER: An Attentional Scene Text Recognizer with Flexible Rectification,” IEEE TPAMI. [IEEE](https://ieeexplore.ieee.org/document/8395027/). Zotero item: `F8HXGKDZ`.
10. Jeonghun Baek et al., “What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis,” arXiv:1904.01906. [arXiv](https://arxiv.org/abs/1904.01906). Zotero item: `53XEAKYC`.

### Local project evidence

- [Current architecture review](./current_architecture_review_packet.md)
- [CPU bottleneck analysis](./eighty_percent_bottleneck_analysis.md)
- [OCR conclusion report](./itda_ocr_conclusion_report.md)
- [Recognition batch ablation](../runs/B0_LOCAL_SCREEN32_REC_BATCH_ABLATION.md)

