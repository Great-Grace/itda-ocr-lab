# Recognition screening protocol

## 목적 분리

GPU는 빠른 정확도 탐색에만 사용한다. CPU 4-thread 수치는 최종 추론 가능성과
정식 리더보드 등재에만 사용한다. GPU와 CPU의 시간을 섞어 하나의 throughput으로
보고하지 않는다.

## Stage A: 폭넓은 GPU 스크리닝

`configs/recognizer_suites/diverse_screen_v1.json`의 25개 조합을 12 epoch로 학습한다.
이 집합은 다음 축을 교차한다.

| 축 | 후보 |
|---|---|
| Visual encoder | plain CNN, MobileNetV3, residual CNN, SVTR |
| Sequence model | 없음, BiLSTM, BiGRU, Transformer encoder |
| Decoder | CTC, autoregressive attention |
| Capacity | hidden/dimension 64, 96, 128; layer 1, 2, 3/4 |

화면 크기, crop 수, split은 두 GPU에서 동일하게 유지한다. 서로 다른 GPU가 같은
모델을 중복 학습하지 않도록 spec 목록을 반으로 나눠 실행한다.

## Stage B: 상위 조합 확인

Stage A에서 CER 상위 6개를 고르고 각 모델을 다음 입력 크기에서 40 epoch까지
재학습한다.

- 256 x 32
- 320 x 48
- 384 x 64

저장 규칙은 Exact Match 우선, Exact 동률이면 CER 최저다. 따라서 `best.pt`가
초기 epoch 가중치로 남지 않는다.

## Stage C: CPU gate와 end-to-end 검증

1. Stage B의 상위 3개에 대해 실제 crop 345개에서 CPU 4-thread latency/CER를 측정한다.
2. CPU 비용이 허용되는 1~2개만 OCR pipeline의 선택적 second-pass로 연결한다.
3. val 459장에 대해 fresh CPU OCR을 실행해 Final EM, Candidate Recall, Selection Accuracy,
   detector/recognizer/postprocess 지연 및 Peak RSS를 기록한다.

GPU 459장 평가 결과는 탐색용 정확도다. 공식 CPU 리더보드에는 CPU fresh run만 올린다.
