#!/usr/bin/env python3
"""Create a review-only Slack-ready benchmark summary. Never sends messages."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "slack_benchmark_draft.md"


def main() -> int:
    lines = [
        "*ITDA OCR Lab — 소비기한 OCR 종합 벤치마크 (검수 초안)*",
        "_아직 Slack 전송 전. 승인 후에만 공유._",
        "",
        "*한 줄 결론*",
        "- 현재 정확도 기준 1순위는 `B4`, CPU 속도·메모리 기준 실용 후보는 `B6`입니다.",
        "- 현재 병목은 날짜를 읽고 후보로 만드는 단계입니다. 날짜 후보가 이미 있으면 고르는 기능은 잘 작동합니다.",
        "",
        "*평가 구성*",
        "- 확정 라벨 이미지 200장: 반복 실험용 `dev100`, 최종 확인용 `lock100`으로 분리",
        "- 최종 추론 조건: CPU-only / 4 threads",
        "- `lock100`은 B4 확정 후보에 대해 한 번만 평가했고, 이후 모델 선택에는 사용하지 않았습니다.",
        "",
        "*주요 결과*",
        "```text",
        "모델 / 구성                               평가셋     정답률   후보생성률  평균시간  최대메모리",
        "B0  PP-OCRv5 mobile + Korean mobile      screen32   34.4%    34.5%      9.76초   1.91GB",
        "B3  B0 + 날짜 형식 보강                   dev100     55.0%    61.5%      9.65초   2.27GB",
        "B4  PP-OCRv5 mobile + PP-OCRv6 medium    dev100     64.0%    69.2%     12.88초   2.61GB",
        "B4  동일 구성, 최종 holdout               lock100    67.0%    71.1%     11.47초   2.17GB",
        "B6  B4 + recognizer 입력 높이 48          dev100     64.0%    70.3%      9.05초   1.95GB",
        "B8  B4 OCR 결과 + 확장 날짜 parser         dev100*    69.0%    75.3%      재인식 없음",
        "```",
        "",
        "*무엇이 효과 있었나*",
        "- 날짜 parser가 `일-월-년`, 영문 월명, 깨진 구분자를 이해하도록 보강 → 후보 생성률 큰 폭 개선",
        "- recognizer를 `PP-OCRv6_medium_rec`으로 변경 → B3 대비 dev100 정답률 +9.0%p",
        "- recognizer 입력 높이 48(B6) → B4와 dev100 정답률은 같고, 약 30% 빠르며 RAM도 감소",
        "- 공백 날짜·영문 월·혼합 구분자 parser(B8) → B4 token cache에서 dev100 정답률 69.0%",
        "",
        "*무엇이 병목인가*",
        "- lock100 오답 33건 중: 날짜 후보 미생성 16건, 글자 인식 오류 10건, 후보 선택 오류 1건",
        "- 후보 선택률 98.4%는 ‘정답 날짜가 후보 목록에 이미 있는 경우’의 조건부 수치입니다.",
        "- 별도 다중 날짜 stress test에서도 20장 중 19장(95.0%)을 맞췄습니다.",
        "- 그래서 다음 단계는 selector 튜닝보다 Recognition/Candidate Generation 개선입니다.",
        "",
        "*제외한 방향*",
        "- grayscale + contrast 전처리: 정확도 하락으로 제외",
        "- selector weight 16개 조합: 실질적 변화 없음",
        "",
        "*다음 실험*",
        "1. 정답 날짜를 강제로 후보 목록에 넣는 Selection stress test",
        "2. 날짜 후보가 없거나 confidence가 낮을 때만 ROI crop + upscale + second-pass",
        "3. 날짜/숫자 전용 recognizer (CRNN/SVTR/attention) — synthetic 학습 수렴 문제를 디버깅 중",
        "4. ICDAR/KAIST 범용 OCR benchmark로 도메인 차이 분리",
        "",
        "*검증 상태*",
        "- `*` B8은 OCR token cache 재해석 실험이므로 처리 시간/메모리는 fresh OCR 결과와 비교하지 않음",
        "- lock100 제출 형식 검사 통과: 100행",
        "- 저장소 테스트: 15 passed",
        "- Slack 전송: 아직 하지 않음",
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
