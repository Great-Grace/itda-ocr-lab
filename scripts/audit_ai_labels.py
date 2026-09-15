#!/usr/bin/env python3
"""Full Census & Quality Audit Harness for AI-generated labels.

Performs exhaustive verification on label datasets without any external paid API calls:
1. Syntactic & Schema Integrity (5 columns, missing values, YYYY-MM-DD format).
2. Calendar Plausibility (detects impossible dates like Feb 30, April 31, month 13).
3. Domain Plausibility (food expiry year ranges, extreme outliers).
4. Ground Truth Cross-Validation (measures exact match, component accuracies, and error types against human GT).
5. OCR Token Physical Existence Check (detects AI hallucinations where date text does not exist in image tokens).
6. High-Confidence Cleansing & Stratification (splits into verified_clean.csv and flagged_suspicious.csv).
7. Automatic Competition Bonus Report Generation (evidence report for the 5-point bonus).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DATE_REGEX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
REQUIRED_COLUMNS = ["image_id", "year", "month", "day", "final_date"]


def parse_date_safe(year_str: str, month_str: str, day_str: str) -> tuple[datetime.date | None, str | None]:
    """Validate calendar date strictly."""
    try:
        y = int(year_str)
        m = int(month_str)
        d = int(day_str)
        return datetime.date(y, m, d), None
    except ValueError as exc:
        return None, str(exc)


def audit_labels(
    labels_path: Path,
    gt_path: Path | None = None,
    tokens_cache_paths: list[Path] | None = None,
    output_dir: Path = Path("runs/label_audit"),
    min_year: int = 2018,
    max_year: int = 2035,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load candidate labels
    rows: list[dict[str, str]] = []
    with open(labels_path, encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            rows.append({k.strip(): v.strip() for k, v in r.items() if k})

    total_count = len(rows)

    # 2. Load ground truth if provided
    gt_dict: dict[str, dict[str, str]] = {}
    if gt_path and gt_path.exists():
        with open(gt_path, encoding="utf-8", newline="") as fp:
            for r in csv.DictReader(fp):
                gt_dict[r["image_id"].strip()] = {k.strip(): v.strip() for k, v in r.items()}

    # 3. Load token caches if provided
    tokens_by_image: dict[str, list[dict[str, Any]]] = {}
    if tokens_cache_paths:
        for p in tokens_cache_paths:
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        img_id = item.get("image_id")
                        if img_id:
                            tokens_by_image[img_id] = item.get("tokens", [])
                    except Exception:
                        pass

    # 4. Audit each row
    clean_rows: list[dict[str, str]] = []
    flagged_rows: list[dict[str, Any]] = []

    stats = {
        "total_images": total_count,
        "valid_dates": 0,
        "none_dates": 0,
        "syntax_errors": 0,
        "calendar_errors": 0,
        "domain_year_outliers": 0,
        "gt_evaluated": 0,
        "gt_exact_matches": 0,
        "gt_mismatches": 0,
        "token_evaluated": 0,
        "token_hallucinations": 0,
        "token_confirmed": 0,
    }

    year_distribution: Counter = Counter()
    error_type_counter: Counter = Counter()

    for row in rows:
        image_id = row.get("image_id", "")
        year = row.get("year", "")
        month = row.get("month", "")
        day = row.get("day", "")
        final_date = row.get("final_date", "")

        flags: list[str] = []

        # Check column existence
        if not image_id:
            flags.append("MISSING_IMAGE_ID")
            stats["syntax_errors"] += 1
            error_type_counter["MISSING_IMAGE_ID"] += 1

        # Check NONE cases
        if final_date == "NONE":
            stats["none_dates"] += 1
            if year != "NONE" or month != "NONE" or day != "NONE":
                flags.append("FINAL_NONE_BUT_COMPONENTS_SET")
                stats["syntax_errors"] += 1
                error_type_counter["FINAL_NONE_BUT_COMPONENTS_SET"] += 1
        else:
            # Check regex format
            m = DATE_REGEX.fullmatch(final_date)
            if not m:
                flags.append("INVALID_DATE_FORMAT")
                stats["syntax_errors"] += 1
                error_type_counter["INVALID_DATE_FORMAT"] += 1
            else:
                fy, fm, fd = m.groups()
                if year != fy or month != fm or day != fd:
                    flags.append(f"COMPONENT_MISMATCH: final({final_date}) != ({year}-{month}-{day})")
                    stats["syntax_errors"] += 1
                    error_type_counter["COMPONENT_MISMATCH"] += 1

                # Calendar validity
                d_obj, cal_err = parse_date_safe(fy, fm, fd)
                if d_obj is None:
                    flags.append(f"CALENDAR_INVALID: {cal_err}")
                    stats["calendar_errors"] += 1
                    error_type_counter["CALENDAR_INVALID"] += 1
                else:
                    stats["valid_dates"] += 1
                    year_val = d_obj.year
                    year_distribution[str(year_val)] += 1
                    if year_val < min_year or year_val > max_year:
                        flags.append(f"YEAR_OUTLIER: {year_val} not in [{min_year}, {max_year}]")
                        stats["domain_year_outliers"] += 1
                        error_type_counter["YEAR_OUTLIER"] += 1

        # GT cross check
        if image_id in gt_dict:
            stats["gt_evaluated"] += 1
            gt = gt_dict[image_id]
            gt_final = gt.get("final_date", "")
            if final_date == gt_final:
                stats["gt_exact_matches"] += 1
            else:
                stats["gt_mismatches"] += 1
                gt_y = gt.get("year", "")
                gt_m = gt.get("month", "")
                gt_d = gt.get("day", "")
                diff_reason = []
                if year != gt_y:
                    diff_reason.append(f"Y({year}!=gt:{gt_y})")
                if month != gt_m:
                    diff_reason.append(f"M({month}!=gt:{gt_m})")
                if day != gt_d:
                    diff_reason.append(f"D({day}!=gt:{gt_d})")
                flags.append(f"GT_MISMATCH: {final_date} vs GT {gt_final} [{', '.join(diff_reason)}]")
                error_type_counter["GT_MISMATCH"] += 1

        # Token physical presence check
        if image_id in tokens_by_image and final_date != "NONE":
            stats["token_evaluated"] += 1
            tokens = tokens_by_image[image_id]
            all_text = " ".join(t.get("text", "") for t in tokens)
            clean_text = re.sub(r"[^\w]", "", all_text)

            # Check if components appear together or separately
            fy = row.get("year", "")
            fm = row.get("month", "")
            fd = row.get("day", "")
            y_short = fy[2:] if len(fy) == 4 else fy

            # Search in concatenated text
            exact_seq = f"{fy}{fm}{fd}"
            exact_short_seq = f"{y_short}{fm}{fd}"
            exact_dmy = f"{fd}{fm}{fy}"
            exact_dmy_short = f"{fd}{fm}{y_short}"

            found_in_tokens = (
                (exact_seq in clean_text)
                or (exact_short_seq in clean_text)
                or (exact_dmy in clean_text)
                or (exact_dmy_short in clean_text)
                or (fy in all_text and fm in all_text and fd in all_text)
            )

            if not found_in_tokens:
                flags.append(f"OCR_HALLUCINATION_OR_MISSING: {final_date} not evidenced in OCR tokens")
                stats["token_hallucinations"] += 1
                error_type_counter["OCR_HALLUCINATION_OR_MISSING"] += 1
            else:
                stats["token_confirmed"] += 1

        # Classify row
        if not flags:
            clean_rows.append(row)
        else:
            flagged_rows.append({
                **row,
                "flags": " | ".join(flags),
                "num_flags": len(flags),
            })

    # 5. Write outputs
    clean_csv_path = output_dir / "verified_clean.csv"
    with open(clean_csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(clean_rows)

    flagged_csv_path = output_dir / "flagged_suspicious.csv"
    flagged_fieldnames = REQUIRED_COLUMNS + ["num_flags", "flags"]
    with open(flagged_csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=flagged_fieldnames)
        writer.writeheader()
        writer.writerows(flagged_rows)

    metrics_json_path = output_dir / "audit_metrics.json"
    audit_summary = {
        "labels_path": str(labels_path),
        "total_images": total_count,
        "clean_count": len(clean_rows),
        "clean_ratio": round(len(clean_rows) / max(1, total_count) * 100, 2),
        "flagged_count": len(flagged_rows),
        "flagged_ratio": round(len(flagged_rows) / max(1, total_count) * 100, 2),
        "statistics": stats,
        "error_breakdown": dict(error_type_counter),
        "year_distribution": dict(sorted(year_distribution.items())),
    }
    with open(metrics_json_path, "w", encoding="utf-8") as fp:
        json.dump(audit_summary, fp, ensure_ascii=False, indent=2)

    report_path = output_dir / "audit_summary.md"
    gt_em_ratio = (
        round(stats["gt_exact_matches"] / max(1, stats["gt_evaluated"]) * 100, 2)
        if stats["gt_evaluated"] > 0
        else "N/A"
    )
    token_conf_ratio = (
        round(stats["token_confirmed"] / max(1, stats["token_evaluated"]) * 100, 2)
        if stats["token_evaluated"] > 0
        else "N/A"
    )

    md_content = f"""# AI 라벨 전수조사(Quality Census & Audit) 결과 보고서

본 보고서는 ITDA 제3회 연합학술제 예선 가산점(최대 5점: "라벨링 규칙 정의 및 데이터셋 확충") 증빙 및 데이터 품질 검증을 위해 작성되었습니다.
**외부 유료 API(OpenAI/Claude 등) 호출 비용 0원(100% 로컬 오프라인/자체 하네스)**으로 전수 조사를 완수하였습니다.

---

## 1. 전수조사 핵심 요약

- **감사 대상 파일**: `{labels_path.name}`
- **전체 검사 이미지 수**: **{total_count:,}장**
- **품질 인증 무결 라벨 (Clean)**: **{len(clean_rows):,}장 ({audit_summary['clean_ratio']}%)**
- **정밀 검토 의심 라벨 (Flagged)**: **{len(flagged_rows):,}장 ({audit_summary['flagged_ratio']}%)**
- **인간 검증 정답(GT) 교차 일치율**: **{gt_em_ratio}%** ({stats['gt_exact_matches']}/{stats['gt_evaluated']})
- **OCR 토큰 물리적 존재 확인율**: **{token_conf_ratio}%** ({stats['token_confirmed']}/{stats['token_evaluated']})

---

## 2. 결함 및 리스크 유형별 통계 (Error Breakdown)

| 결함 유형 (Flag Type) | 발생 건수 | 설명 및 영향 |
|---|---:|---|
| **CALENDAR_INVALID** | {stats['calendar_errors']} | 2월 30일, 13월 등 달력상 불가능한 날짜 |
| **SYNTAX_ERRORS** | {stats['syntax_errors']} | 포맷 규격 위반, 패딩 누락, 컬럼 불일치 |
| **YEAR_OUTLIER** | {stats['domain_year_outliers']} | 식품 유통기한 범주([{min_year}, {max_year}]) 외 극단 연도 (OCR 오인) |
| **GT_MISMATCH** | {stats['gt_mismatches']} | 검증된 200장 Ground Truth와 불일치 (제조일 오인 등) |
| **OCR_HALLUCINATION_OR_MISSING** | {stats['token_hallucinations']} | 라벨 날짜가 실제 이미지 OCR 텍스트 내에 물리적으로 없음 |

---

## 3. 연도별 분포 (Year Distribution)

| 연도 | 건수 | 비율 |
|---|---:|---:|
"""
    for yr, cnt in sorted(year_distribution.items()):
        pct = round(cnt / max(1, stats["valid_dates"]) * 100, 1)
        md_content += f"| **{yr}** | {cnt:,} | {pct}% |\n"

    md_content += f"""
---

## 4. 정제 규칙 및 후속 조치 (Quality Cleansing SOP)

1. **Clean 라벨 승격 (`verified_clean.csv`)**:
   - 달력 무결성, 식품 연도 유효 범위, OCR 토큰 존재 증거가 모두 확보된 데이터만 최종 모델 학습/검증 세트로 편입.
2. **Flagged 라벨 격리 (`flagged_suspicious.csv`)**:
   - 결함 사유(flags)가 명시된 의심 데이터는 노이즈 유입 방지를 위해 학습 셋에서 자동 제외.
   - 필요 시 상위 신뢰도 규칙(RULE-01, RULE-03)을 재적용하여 제조일/유통기한 우선순위 교정.
3. **규정 준수 및 무과금 보장**:
   - 외부 클라우드 API 호출 0건, 전 과정 오픈소스 및 로컬 파이프라인 수행.
"""
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write(md_content)

    return audit_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Full Census & Quality Audit for AI Labels.")
    parser.add_argument("--labels", required=True, help="Path to input labels CSV")
    parser.add_argument("--gt-labels", default="runs/B0_test_v1/labels.csv", help="Human-verified Ground Truth CSV")
    parser.add_argument("--tokens-cache", action="append", help="OCR tokens jsonl cache paths")
    parser.add_argument("--output-dir", default="runs/label_audit", help="Output directory")
    parser.add_argument("--min-year", type=int, default=2018)
    parser.add_argument("--max-year", type=int, default=2035)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"[ERROR] Labels file not found: {labels_path}")
        return 1

    gt_path = Path(args.gt_labels) if args.gt_labels else None
    tokens_paths = [Path(p) for p in args.tokens_cache] if args.tokens_cache else None

    # Collect default caches if none provided
    if not tokens_paths:
        tokens_paths = list(Path("runs").glob("**/ocr_tokens.jsonl"))

    print(f"=== [AI Label Full Census & Quality Audit] ===")
    print(f"Target: {labels_path}")
    print(f"GT: {gt_path} (exists: {gt_path.exists() if gt_path else False})")
    print(f"Token Caches: {len(tokens_paths) if tokens_paths else 0} files found")

    res = audit_labels(
        labels_path=labels_path,
        gt_path=gt_path,
        tokens_cache_paths=tokens_paths,
        output_dir=Path(args.output_dir),
        min_year=args.min_year,
        max_year=args.max_year,
    )

    print(f"\nAudit Finished!")
    print(f"  - Total: {res['total_images']} rows")
    print(f"  - Clean: {res['clean_count']} ({res['clean_ratio']}%) -> {args.output_dir}/verified_clean.csv")
    print(f"  - Flagged: {res['flagged_count']} ({res['flagged_ratio']}%) -> {args.output_dir}/flagged_suspicious.csv")
    print(f"  - Summary: {args.output_dir}/audit_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
