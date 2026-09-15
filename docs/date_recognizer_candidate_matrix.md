# Date-only recognizer 후보 매트릭스

문자 집합은 기본적으로 다음으로 제한한다.

```text
0123456789.-/:년월일
```

키워드(소비기한, 유통기한, EXP 등)는 primary Korean OCR이 담당하고 date-only
recognizer에는 넣지 않는다.

## 후보

| ID | 구조 | 학습 | 추론 | 목적 |
|---|---|---|---|---|
| D0 | pretrained PP-OCRv5/v6 + parser allowlist | 없음 | CTC 원출력 후 보정 | 비용 0인 기준선 |
| D1 | pretrained backbone + date-only CTC head | head 재학습 → low-LR fine-tune | 병렬 CTC | 주력 정확도/속도 후보 |
| D2 | pretrained backbone + date-only Attention head | head 재학습 → low-LR fine-tune | max length 12 autoregressive | 짧은 날짜의 문맥 보완 후보 |
| D3 | CTC 1차 → low-confidence crop만 Attention | 두 head 공동 학습 | CTC fast path + Attention fallback | 정확도-속도 절충 |
| D4 | SVTRv2식 CTC + multi-size resize/feature rearrangement | teacher-guided fine-tune | CTC only | 기울기·불규칙 crop 대응 |
| D5 | date-only CTC + finite-state grammar decoder | CTC fine-tune | CTC + 날짜 문법 제약 | 13월/31일/잘못된 separator 억제 |

## Attention 비용 판단

문자 수를 줄이면 softmax/classifier 계산량은 줄어들지만, Attention은 여전히
문자마다 decoder step을 실행한다. 따라서:

- 전체 3,352장에 Attention: CPU latency 부담이 큼
- 1차 CTC가 실패한 10~20% crop에만 Attention: 현실적인 후보
- 날짜 길이를 6~12 step으로 제한: 일반 OCR보다 부담이 작음

정확한 비용은 같은 crop 345개에서 median/p95를 직접 잰다. Attention이 CTC보다
느려도 fallback 비율이 15% 이하이고 Final EM이 2%p 이상 오르면 채택한다.

## 학습 데이터 조건

- 실제 최종 detector와 동일한 crop geometry
- 정답과 이미 일치한 쉬운 crop만 사용하지 않음
- 도트매트릭스·저대비·분할 인식·기울어진 crop을 hard set으로 포함
- train/val image ID 완전 분리
- crop CER가 아니라 459장 end-to-end EM을 승격 기준으로 사용

## GPU 실행 순서

1. D1 date-only CTC
2. D2 date-only Attention
3. D3 CTC+Attention fallback
4. D4/D5는 D1/D3가 유효할 때만 확장

각 후보는 crop accuracy, Final EM, Candidate Recall, p50/p95 CPU latency, Peak
RSS를 같은 표에 기록한다.
