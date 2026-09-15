# 제3회 ITDA 연합학술제 공식 규정 및 채점 가이드

본 문서는 ITDA 연합학술제 공식 참가자 안내서의 기술 규정 및 채점 기준을 정리한 레퍼런스입니다.

---

## 1. 운영진 채점 환경
- **CPU 환경**: Standard 4-Core vCPU (GPU 미제공, CPU 전용 채점).
- **네트워크**: **완전 차단된 오프라인 환경**.
  - 런타임 중 인터넷 호출 또는 모델 가중치 자동 다운로드 시 **즉시 실격 / 0점**.
  - 유료 LLM/VLM 외부 API (GPT-4o, Claude 등) 호출 절대 불가.
- **Python 버전**: Python 3.10 기준.
- **의존성**: `requirements.txt`에 명시된 패키지만 자동 설치 (반드시 `nbconvert`, `ipykernel` 포함).
- **타임아웃**: 전체 실행 시간 2,400초 (40분) 제한.

---

## 2. 제출 규격 및 필수 파일 구조
```text
itda3-{학회영문}-{영문팀명}/
├── predict.ipynb       # 메인 추론 노트북 (운영진 채점용, 필수 단일 채점 파일)
├── requirements.txt    # 실행 환경 패키지 목록 (필수)
├── README.md           # 가중치 다운로드 및 실행 가이드 문서 (필수)
├── .gitignore          # 가중치·데이터 커밋 방지 (필수)
├── download_weights.sh # 외부 가중치 다운로드 스크립트 (필수)
└── weights/            # 모델 가중치 저장 폴더
```

### `predict.ipynb` 필수 규격
1. **첫 번째 코드 셀 (설정 셀 필수 형식)**:
   ```python
   # ===== CONFIG =====
   import os
   INPUT_DIR = os.environ.get("ITDA_INPUT_DIR", "./val_images")
   OUTPUT_PATH = os.environ.get("ITDA_OUTPUT_PATH", "./submission.csv")
   # ==================
   ```
2. **실행 방식**: Run All 순차 실행 (사용자 입력 요구 금지).
3. **최종 출력**: `OUTPUT_PATH` 경로에 `submission.csv` 저장.
   - `df.to_csv(OUTPUT_PATH, index=False)` (인덱스 반드시 제외).
4. **출력 스키마 규격**:
   - 컬럼: `image_id,year,month,day,final_date`
   - 미인식 시: `NONE`
   - 예시:
     ```csv
     image_id,year,month,day,final_date
     1,2026,05,29,2026-05-29
     2,NONE,NONE,NONE,NONE
     ```

---

## 3. 평가 배점 및 전략 (1차 예선 100점 + 가산점 5점)
- **정량 평가 (60점)**:
  - 날짜 추출 정확도: 50점
  - CPU 추론 속도: 10점 (4-Core vCPU 최적화 필수)
- **정성 평가 (40점)**:
  - 아키텍처 구성 및 논리성 (요약서 1p): 15점
  - 설계 논리 및 최적화/활용 전략 (요약서 2p): 25점
- **가산점 (최대 5점)**:
  - **데이터 직접 수집/라벨링**:
    - 참가자가 직접 라벨링 규칙을 정의하고 데이터를 수집/확장/라벨링 (요약서 2p 증빙 필수).
    - FAQ 지침: "3,352장을 다 사람이 라벨링할 필요는 없으며, 정규식/OCR 기반 룰로 자동 라벨을 생성하거나 예외 케이스를 보완하는 주체적 전략 활용".
