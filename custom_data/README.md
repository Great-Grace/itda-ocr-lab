# Custom Data for Expiration Date OCR (가산점 증빙)

본 디렉터리는 제3회 ITDA 연합학술제 예선 가산점(최대 5점: "라벨링 규칙을 직접 정의하고 데이터를 직접 수집/확장/라벨링") 평가를 위한 자체 구축 및 전수 감사 데이터셋입니다.

---

## 1. 폴더 구조
```text
custom_data/
├── README.md            # 본 안내 문서 (데이터 위치 및 구조 설명)
├── labeling_rules.md    # 자체 라벨링 규칙 정의서 (RULE-01 ~ RULE-04)
├── labels.csv           # 정답 라벨 파일 (image_id, filename, year, month, day, final_date, confidence, text_found, reason)
└── images/              # 자체 수집 및 난이도별(도트매트릭스, 난반사, 캡인쇄) 원본 이미지 (80장, 총 18MB)
```

## 2. 라벨 파일 명세 (`labels.csv`)
- **image_id**: 이미지 고유 번호 (6자리)
- **filename**: 이미지 파일명 (`002334.jpg` 등)
- **year / month / day**: 파싱된 연/월/일 (2자리 0-padding, 결측 시 `NONE`)
- **final_date**: `YYYY-MM-DD` 형식 (결측 시 `NONE`)
- **confidence**: 라벨링 신뢰도 (`high`, `medium`)
- **text_found**: 이미지 내 실제 검출된 원문 텍스트 (예: `소비기한 2027.06.26 까지`, `2024.10.15까지`)
- **reason**: 해당 일자를 소비기한으로 판정한 근거 및 라벨링 논리

## 3. 구축 기준 및 품질 관리
- **도메인 특화 샘플링**: 일반 인쇄 텍스트 외에 OCR 실패율이 높은 잉크젯 도트매트릭스(Continuous Inkjet), 병뚜껑 캡 스탬프, 음각/양각 캔 하단 인쇄, 비닐 파우치 굴곡면을 집중 선별.
- **다단계 교차 검증**: 외부 유료 API에 의존하지 않고 로컬 OCR 다중 엔진 교차 검증 + 캘린더 정합성 검증 + 100% 휴먼 전수 감사(Human-in-the-loop audit)를 통해 무결성을 확보함.
