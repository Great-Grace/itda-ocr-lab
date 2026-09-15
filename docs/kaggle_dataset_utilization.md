# Kaggle Korean Expiry OCR Dataset: verification and use plan

검증일: 2026-09-13

## 확인된 구조

공개 Kaggle 데이터셋 `kimhyeongminkhu/korean-expiry-date-ocr`의 v3 파일 목록을 API로 전수 열람했다(23,386 files, 약 2.73 GB). 원본 이미지 전체를 로컬로 받지 않고 ZIP 중앙 디렉터리와 필요한 샘플만 범위 요청으로 확인했다.

| Subset | Classes | 역할 | 이미지 수(목록 기준) |
| --- | --- | --- | ---: |
| `expiry_region_detection_dataset` | `date`, `due`, `code`, `full` | 제품 전체 이미지에서 날짜/유통기한/코드/전체 날짜 영역 검출 | train 3,353, val 808 |
| `expiry_date_detection_dataset` | `date`, `due`, `code` | 날짜 영역 crop에서 날짜·유통기한·코드 검출 | train 3,354, val 810 |
| `expiry_dmy_detection_dataset` | `Y4`, `ENG_MON`, `NUM2` | 날짜 crop에서 연도·영문 월·숫자(월/일) 영역 검출 | train 2,894, val 478 |

주석은 모두 YOLO 형식의 class/center-x/center-y/width/height box다. 따라서 이 데이터는 단순 OCR 문자열 라벨이 아니라 detector를 직접 학습할 수 있는 지역/구성요소 주석을 포함한다.

## 우리 데이터와의 동일성 확인

`expiry_region_detection_dataset`의 파일을 Kaggle ZIP에서 범위 요청으로 읽어 기존 Drive box-audit 이미지와 CRC32/SHA-256을 비교했다. box-audit에 포함된 80장 모두 Kaggle 원본과 정확히 매칭되었고, 003181은 SHA-256까지 일치했다.

```text
Kaggle region 003181.jpg: 0db96ee15791e4aaef870fbbe1069c3b7ff85a5398d8aab024d073e7ea74ba73
기존 Drive 003181.jpg:    0db96ee15791e4aaef870fbbe1069c3b7ff85a5398d8aab024d073e7ea74ba73
```

즉 region subset은 우리가 사용한 상품 이미지와 동일한 원본을 포함한다. 반면 `expiry_date`는 183x67 같은 날짜 crop 이미지이고, Kaggle 전체에는 원본 3,352장 외에 추가/합성 샘플이 섞여 있으므로 파일 개수를 그대로 우리 대회 데이터 개수로 간주하면 안 된다. 학습/평가는 반드시 파일명·해시 기준으로 분리해야 한다.

80장 box-audit 매칭 결과의 region box 구성은 `date` 90개, `due` 46개, `code` 66개, `full` 81개였다. 한 이미지에 여러 종류의 box가 공존하므로 `full` 또는 `date/due`를 목적별로 선택하는 다중 헤드/단계식 detector 실험이 가능하다.

## 권장 활용

1. **1차 detector 실험**: `expiry_region`의 `date`/`due`/`full` box를 사용해 날짜 영역 detector를 학습한다. `code`는 hard-negative 또는 보조 클래스로 유지한다.
2. **2차 date-crop detector**: `expiry_date`의 `date`/`due` box로 crop 내부의 유효 날짜 위치를 정제한다.
3. **3차 구성요소 보조 실험**: `expiry_dmy`의 `Y4`/`ENG_MON`/`NUM2` box는 바로 문자 인식 정답이 아니므로 recognizer label로 오인하지 않는다. 구성요소 detector 또는 crop-guided attention/CTC의 보조 supervision으로만 사용한다.
4. **누수 방지**: 우리 3,063장(Gemini-filtered)과 Kaggle 원본/추가 샘플을 해시·파일명으로 교집합 계산한 뒤, 대회 val/test와 동일 이미지가 학습에 들어가지 않도록 고정한다. 대회 성능 보고에는 `kaggle_external`, `itda_gemini_filtered` provenance를 분리한다.
5. **GPU 실행 순서**: (a) region date/full detector, (b) detector crop + 기존 PP-OCRv6 recognizer, (c) crop recognizer/Attention 후보 순으로 smoke → 50~100장 sanity → val 실측을 한다. GPU는 이 교집합 검증과 변환이 끝난 뒤에만 켠다.

## 현재 결론

Kaggle 데이터셋은 활용 가치가 높다. 특히 지금까지의 병목이 full-image detector와 recognizer crop 불일치였기 때문에, `expiry_region`의 실제 box와 `expiry_date`의 정제 crop을 함께 쓰면 **detector-guided recognizer 학습**을 재현할 수 있다. 다만 Kaggle 카드의 96% 수치는 그대로 채택하지 않고, 우리 고정 split에서 독립적으로 측정한다.
