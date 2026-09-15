# ITDA 학술제 팀 DM / Claude artifact 교차검토

검토 범위: Slack 그룹 DM `C0BTL61PZ8B`에서 공유된 Claude artifact 5종과 해당 DM의 후속 논의.

## 팀원 실험에서 재사용할 수 있는 실측 사실

| 출처 | 방법 | 결과/의미 |
|---|---|---|
| OCR 실험 보고서 `6679…` | CPU 100장 단계별 분해 | Detection 478ms(27.5%), Crop 151ms(8.7%), Recognition 1,112ms(63.9%) |
| `6679…` | box priority: aspect ratio 내림차순 + cap | 평균 인식 box 19.8→9.6, recognition 1,218→591ms. 단 000001/000007처럼 box가 많은 이미지에서 정답 손실 확인 |
| `6679…` | thread sweep | 4코어가 1,745ms(1코어), 1,783ms(2), 1,408ms(4), 1,910ms(8), 2,098ms(12)로 최적 |
| `6679…` | rectification | 방향 보정이 87% 악화, UVDoc 4,776ms/장으로 제외 |
| `6679…` | CLAHE 조건부 재시도 | NONE 7장 중 1장 해결, 전체 적용은 +68% 비용. 조건부 retry만 타당 |
| `a2d244…` | PaddleOCR v5 mobile + 규칙/CLAHE | 150장 dev 51.3→58.7→60.7→64.0→72.67%, 최종 109/150 |
| `a2d244…` | v3/v4/v5 detector | 150장 기준 v3 54.67%, v4 57.33%, v5 64.00%; 이 조건에서는 v5 유지 |
| `a2d244…` | RapidOCR 단독 | Paddle과 동일 selector에서 64.0%, 평균 3.49s vs 14.3s, p95 5.67s vs 51.5s |
| `a2d244…` | RapidOCR + CLAHE | 69.33% (104/150), Paddle + CLAHE 72.67% (109/150). 정확도는 Paddle 우세, 속도는 Rapid 우세 |
| `eb808…` | RapidOCR ONNX + cascade | 3,352장 중 2,560장 날짜 추출(76.4%), p50 3.61s, p95 13.4s, 40장 검증 85%. NONE 표본 대부분 도트매트릭스 숫자 |
| `eb808…` | cascade | raw → 도트매트릭스 CLAHE/blur/morphology → 2000px → 2400px 재검출. 1차 2,913장, retry 439장 |
| `352e…` | 문헌 조사 | 만료일 OCR 선행연구는 대부분 “날짜 영역 검출→crop recognizer” 2단계. 도트매트릭스는 일반 OCR과 다른 인쇄 형태 |
| `09f191…` | EasyOCR 9장 실측 | 640px에서도 약 2.11s/장, 3,352장 환산 118분. 라이브러리 교체보다 full-image detection 축소가 핵심 |

## 우리 실험과의 교집합

- 우리 GPU val: PP-OCRv5 mobile EM 55.74%, v6 medium EM 70.16%, Candidate Recall 75.74%, Selection 92.64%.
- 팀 Rapid cascade: Candidate Coverage를 명시하진 않았지만 추출률 76.4%, 추출 성공 표본 정밀도 약 95%로 보고했다.
- 두 결과 모두 “후보가 생긴 뒤 선택”은 상대적으로 강하고, 실패 대부분은 숫자 영역의 검출/인식 실패다.
- 팀의 4코어 최적점, 회전 보정 기각, 조건부 CLAHE, OCR cache 재선택은 우리 측정 방향과 일치한다.

## 혼합 아키텍처 후보

`Detector/Recognizer → Candidate Generation → Rule Selector`를 한 덩어리로 비교하지 않고 다음을 독립 측정한다.

1. **정확도 우선**: PP-OCRv5 mobile det + PP-OCRv6 medium rec + keyword/spatial selector + NONE-only CLAHE retry.
2. **시간 우선**: RapidOCR ONNX PP-OCRv5 + 동일 selector + 동일 NONE-only retry.
3. **혼합 cascade**: RapidOCR 1차 → 날짜 후보가 없거나 confidence가 낮을 때만 Paddle/v6 crop second-pass.
4. **recognizer 연구**: 날짜 crop이 확보된 경우 숫자 전용 SVTR/Attention/CRNN을 second-pass로 제한 적용.
5. **검출 연구**: 충분한 위치 라벨이 생길 때만 YOLOv8n/DBNet+CBAM 날짜영역 detector를 검토한다. 현재 라벨이 없으므로 즉시 주력으로 삼지 않는다.

## 다음 실험 게이트

- 같은 459장 val, 같은 labels/split, 같은 selector/normalizer로 RapidOCR과 Paddle/v6를 비교한다.
- GPU는 정확도 스크리닝, CPU 4-thread는 latency/RSS 공식 측정으로 분리한다.
- `Candidate Recall`, `Selection Accuracy | Candidate Exists`, `p50/p95`, `NONE rate`를 반드시 함께 기록한다.
- 40장/150장 팀 dev 결과는 참고용으로 보존하되, 우리 공식 리더보드에는 459장 fresh CPU run만 등재한다.

## 우리 환경에서 재실측한 교차 비교

2026-09-13 VESSL, 동일한 `data/splits/val.csv` 459장과 동일한 규칙 기반 selector로 재측정했다.

| Backend | Final EM | Candidate Recall | Selection Acc. | CPU/GPU sec/image | Peak RAM |
|---|---:|---:|---:|---:|---:|
| PP-OCRv5 mobile GPU | 55.74% | 57.05% | 97.70% | GPU 0.248 | 1.86GB |
| PP-OCRv6 medium GPU | 70.16% | 75.74% | 92.64% | GPU 0.395 | 1.84GB |
| RapidOCR ONNX + cascade CPU 4T | 64.92% | 67.87% | 95.65% | CPU 0.977 | 328MB |

동일 RapidOCR run에 저비용 true CLAHE retry를 붙인 재실측은 EM 65.25%, Candidate Recall 68.20%, Selection 95.67%, 0.942초/장, Peak RSS 303MB였다. 개선폭은 +0.33%p로 작아, CLAHE는 보조 옵션으로 두고 숫자 전용 recognizer/날짜 영역 검출을 우선한다.

해석: RapidOCR은 팀원의 속도 방향을 재현했지만, 현재 459장 기준 정확도는 v6보다 낮다. 따라서 RapidOCR은 시간 제약용 후보, Paddle v6는 정확도 후보로 분리하고, 최종 선택은 동일 CPU 4-thread에서 v6를 끝까지 측정한 뒤 결정한다.
