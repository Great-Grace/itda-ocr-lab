# ITDA OCR Lab — Agent Operational Rules & SOP

이 문서는 Gemini, Claude Code, GitHub Copilot/Codex, Antigravity 등 모든 AI 에이전트가 공통으로 준수해야 하는 **단일 진실 공급원(Single Source of Truth)** 지침입니다.

---

## 1. 아키텍처 및 모듈화 원칙 (스파게티 절대 방지)

* **파이프라인 불변성 (Config-First)**:
  * 실험을 변경할 때 `src/ocr_lab/pipeline.py`를 함부로 수정하지 마세요. 모든 실험의 변화는 `configs/experiments/<이름>.yaml`의 파라미터로 제어되어야 합니다.
* **PyTorch 스타일의 독립 모듈화**:
  * 모든 기능은 `src/ocr_lab/modules/` 아래에 독립된 클래스로 구현합니다.
  * 생성자 `__init__(**params)`는 하이퍼파라미터를 받고, 인풋과 아웃풋은 `src/ocr_lab/contracts.py`의 규격을 엄격히 따릅니다:
    * `preprocess`: `(image_path, output_dir) -> preprocessed_image_path`
    * `ocr`: `image_path -> list[OCRToken]`
    * `selector`: `list[OCRToken] -> DateCandidate | None`
    * `normalizer`: `(image_id, DateCandidate) -> Prediction`
  * 신규 모듈을 만들면 `src/ocr_lab/modules/__init__.py` 레지스트리에만 등록하고, 파이프라인의 다른 부분은 건드리지 마세요.
* **토큰 캐싱 활용**:
  * 가중치나 정규식, 필터링 등 `selector`/`normalizer`만 변경하는 실험의 경우, 무거운 OCR 추론을 반복하지 말고 `--tokens-cache <이전 ocr_tokens.jsonl>` 플래그를 활용하세요.

---

## 2. 비개발자 자연어 요청 처리 SOP (5단계 워크플로우)

비개발자가 *"유통기한 인식률 높여줘"*, *"기울어진 사진 보정해봐"* 처럼 일상어로 요청하면 다음 5단계를 자율적으로 수행하세요:

1. **가설 및 변경 대상 식별**:
   * 어떤 모듈(`preprocess`, `ocr`, `selector`, `normalizer`)을 어떻게 바꿀지 판단합니다.
   * 필요시 [docs/intake_schema.md](docs/intake_schema.md)를 참고해 최소한의 핵심 질문만 사용자에게 던집니다.
2. **새로운 Config 생성**:
   * `configs/experiments/<실험명>.yaml` 파일을 생성합니다. (기존 베이스라인을 직접 덮어쓰지 마세요)
3. **1초 스모크 테스트 (필수)**:
   * 큰 실험이나 GPU 작업을 돌리기 전에, 먼저 내장 샘플로 무결성을 검증합니다:
     ```bash
     python scripts/run_experiment.py \
       --config configs/experiments/<실험명>.yaml \
       --input data/sample \
       --output runs/smoke_check \
       --labels data/sample/labels.csv
     ```
4. **본 실험 및 검증 게이트 통과**:
   * 본 데이터셋으로 실행 후 반드시 제출물 스키마를 검증합니다:
     ```bash
     python scripts/check_submission.py runs/<실험명>/predictions.csv
     ```
   * GPU 모델의 경우, 최종 후보는 반드시 `--device cpu` 환경에서도 에러 없이 동작해야 합니다.
5. **비개발자 맞춤형 쉬운 보고**:
   * 기술적인 로그나 거대한 CSV 대신, 자동 생성된 `runs/<실험명>/summary.md` 및 `runs/<실험명>/review.html`을 바탕으로 결과를 설명합니다:
     * **정확도 변화**: (예: 66.7% -> 100% 상승)
     * **처리 속도**: 평균 몇 ms 소요
     * **실패 사례 분석**: 오답이 발생한 이미지와 그 이유(예: "소비기한 글자가 잘려 인식 실패")

---

## 3. 엄격한 금지 사항 (Hard Rules)

- ❌ 최종 추론 코드(`predict.ipynb`) 및 제출 파이프라인에서 인터넷/외부 API를 호출하거나 모델 가중치를 다운로드하지 마세요.
- ❌ 원본 이미지, 대회 라벨, 가중치 바이너리, 비밀키/토큰, `runs/` 결과물을 Git에 커밋하지 마세요.
- ❌ `python scripts/check_submission.py` 검증 통과 없이 성공을 선언하지 마세요.
