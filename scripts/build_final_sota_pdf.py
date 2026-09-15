from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/itda_ocr_current_sota_report.pdf"
FONT_REG = Path("/Users/taewoo/Library/Fonts/AppleSDGothicNeoR.ttf")
FONT_BOLD = Path("/Users/taewoo/Library/Fonts/AppleSDGothicNeoB.ttf")
pdfmetrics.registerFont(TTFont("AppleSDG", str(FONT_REG)))
pdfmetrics.registerFont(TTFont("AppleSDGB", str(FONT_BOLD)))

PAGE_W, PAGE_H = A4
NAVY = colors.HexColor("#132238")
BLUE = colors.HexColor("#2F6BFF")
TEAL = colors.HexColor("#00A88F")
MINT = colors.HexColor("#E9F8F4")
PALE_BLUE = colors.HexColor("#EEF3FF")
INK = colors.HexColor("#1E293B")
MUTED = colors.HexColor("#64748B")
LINE = colors.HexColor("#D7DEE8")
RED = colors.HexColor("#B42318")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="KTitle", parent=styles["Title"], fontName="AppleSDGB", fontSize=26, leading=32, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8))
styles.add(ParagraphStyle(name="KSubtitle", parent=styles["Normal"], fontName="AppleSDG", fontSize=11, leading=17, textColor=MUTED, spaceAfter=12))
styles.add(ParagraphStyle(name="H1K", parent=styles["Heading1"], fontName="AppleSDGB", fontSize=17, leading=22, textColor=NAVY, spaceBefore=4, spaceAfter=9))
styles.add(ParagraphStyle(name="H2K", parent=styles["Heading2"], fontName="AppleSDGB", fontSize=12.5, leading=17, textColor=BLUE, spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle(name="BodyK", parent=styles["BodyText"], fontName="AppleSDG", fontSize=9.2, leading=14.2, textColor=INK, spaceAfter=5, wordWrap="CJK"))
styles.add(ParagraphStyle(name="SmallK", parent=styles["BodyText"], fontName="AppleSDG", fontSize=7.8, leading=11.2, textColor=MUTED, spaceAfter=3, wordWrap="CJK"))
styles.add(ParagraphStyle(name="TableK", parent=styles["BodyText"], fontName="AppleSDG", fontSize=7.6, leading=10, textColor=INK, wordWrap="CJK"))
styles.add(ParagraphStyle(name="TableKB", parent=styles["BodyText"], fontName="AppleSDGB", fontSize=7.6, leading=10, textColor=INK, wordWrap="CJK"))
styles.add(ParagraphStyle(name="TableHeaderK", parent=styles["BodyText"], fontName="AppleSDG", fontSize=7.6, leading=10, textColor=NAVY, wordWrap="CJK"))
styles.add(ParagraphStyle(name="CalloutK", parent=styles["BodyText"], fontName="AppleSDGB", fontSize=10.5, leading=15.5, textColor=NAVY, wordWrap="CJK"))
styles.add(ParagraphStyle(name="CalloutWhiteK", parent=styles["BodyText"], fontName="AppleSDGB", fontSize=9.4, leading=13.5, textColor=colors.white, wordWrap="CJK"))


def P(text: str, style: str = "BodyK") -> Paragraph:
    return Paragraph(text, styles[style])


def pct(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def cell(text: str, bold: bool = False) -> Paragraph:
    return P(str(text), "TableKB" if bold else "TableK")


def styled_table(data, widths, header=True, row_heights=None):
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            if not isinstance(value, Paragraph):
                data[i][j] = P(str(value), "TableK")
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, rowHeights=row_heights)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), PALE_BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), NAVY)]
        for j in range(len(data[0])):
            value = data[0][j].getPlainText() if isinstance(data[0][j], Paragraph) else str(data[0][j])
            data[0][j] = P(value, "TableHeaderK")
        commands += [("BACKGROUND", (0, 1), (-1, -1), colors.white)]
        for i in range(1, len(data)):
            if i % 2 == 0:
                commands.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F8FAFC")))
    table.setStyle(TableStyle(commands))
    return table


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 14 * mm, PAGE_W - 18 * mm, 14 * mm)
    canvas.setFont("AppleSDG", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "ITDA OCR Lab | validation evidence report")
    canvas.drawRightString(PAGE_W - 18 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def load_data():
    full = json.loads((ROOT / "runs/colab/colab_full_baseline_v6_val459_correct_metrics.json").read_text())
    yolo = json.loads((ROOT / "runs/colab/t4_yolo_v6_accuracy_459.json").read_text())
    union = json.loads((ROOT / "runs/colab/t4_full_yolo_union_trainfit_selector.json").read_text())["evaluation"]
    return full, yolo, union


def build():
    full, yolo, union = load_data()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=16 * mm, bottomMargin=19 * mm, title="ITDA OCR 현재 SOTA 검증 보고서", author="ITDA OCR Lab")
    story = []

    # Cover
    story.append(Spacer(1, 13 * mm))
    story.append(P("ITDA OCR Lab", "SmallK"))
    story.append(P("현재 SOTA 모델\n대회 환경 검증 보고서", "KTitle"))
    story.append(P("소비기한 날짜 추출 | PP-OCRv6 + expiry detector union", "KSubtitle"))
    hero = Table([[P("80.61%", "KTitle"), P("Final Exact Match\n459장 validation 실측", "CalloutK")]], colWidths=[58 * mm, 92 * mm], rowHeights=[27 * mm])
    hero.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE), ("BOX", (0, 0), (-1, -1), 0.8, BLUE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12)]))
    story.append(hero)
    story.append(Spacer(1, 8 * mm))
    story.append(P("핵심 결론", "H1K"))
    story.append(P("전체 이미지 OCR을 유지하면서 YOLO expiry crop 후보를 추가하고, recognition confidence와 bbox 공간 겹침을 함께 사용하는 union selector가 현재 가장 높은 실측 성능을 냈습니다. YOLO crop만 단독 사용하면 성능이 오히려 떨어지므로, YOLO는 전체 OCR을 대체하는 모델이 아니라 후보를 보강하는 branch로 운용해야 합니다.", "CalloutK"))
    story.append(Spacer(1, 4 * mm))
    summary = [
        [cell("평가 대상", True), cell("ITDA validation 459장 (val.csv ID 명시 고정)")],
        [cell("학습 데이터", True), cell("ITDA Gemini-filtered label + Kaggle 동일 원본 box supervision")],
        [cell("test 사용 여부", True), cell("사용하지 않음")],
        [cell("권장 운영", True), cell("YOLO 1차 후보 생성 -> 저신뢰/문맥 부족 샘플만 full OCR fallback")],
    ]
    story.append(styled_table(summary, [35 * mm, 115 * mm], header=False))
    story.append(Spacer(1, 4 * mm))
    story.append(P("작성일: 2026-09-14 | 본 문서는 내부 validation 근거 보고서이며 외부 학술 SOTA를 주장하지 않습니다.", "SmallK"))
    story.append(PageBreak())

    # Architecture and protocol
    story.append(P("1. 최종 아키텍처", "H1K"))
    flow = [[P("원본 이미지", "CalloutK"), P("->", "CalloutK"), P("Branch A\nPP-OCRv5 mobile det\n+ PP-OCRv6 medium rec", "TableK"), P("+", "CalloutK"), P("Branch B\nYOLOv8n expiry binary\n+ PP-OCRv6 crop rec", "TableK"), P("->", "CalloutK"), P("union 후보\nspatial selector\n정규화 출력", "TableK")]]
    flow_table = Table(flow, colWidths=[25 * mm, 8 * mm, 35 * mm, 8 * mm, 35 * mm, 8 * mm, 31 * mm], rowHeights=[24 * mm])
    flow_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#F8FAFC")), ("BACKGROUND", (2, 0), (2, 0), PALE_BLUE), ("BACKGROUND", (4, 0), (4, 0), MINT), ("BACKGROUND", (6, 0), (6, 0), colors.HexColor("#FFF7E6")), ("BOX", (0, 0), (-1, -1), 0.6, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (1, 0), "CENTER"), ("ALIGN", (3, 0), (3, 0), "CENTER"), ("ALIGN", (5, 0), (5, 0), "CENTER"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
    story.append(flow_table)
    story.append(Spacer(1, 4 * mm))
    story.append(P("Branch A는 전체 문맥을 보존해 키워드와 제조일/소비기한 관계를 파악합니다. Branch B는 Kaggle의 date/due box로 학습한 expiry detector가 날짜 주변을 정밀하게 제안합니다. 두 후보를 합친 뒤 train-only로 고정한 selector가 최종 날짜를 선택합니다.", "BodyK"))
    story.append(P("selector weight", "H2K"))
    selector_data = [["source bonus", "candidate score", "recognition log", "detection log", "bbox IoU"], ["YOLO 1.5", "1.5", "1.5", "0.5", "0.5"]]
    story.append(styled_table(selector_data, [28 * mm, 34 * mm, 34 * mm, 30 * mm, 30 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(P("평가 프로토콜", "H2K"))
    protocol = [[cell("항목", True), cell("설정", True)], [cell("입력", True), cell("Raw product image, 전역 전처리 없음")], [cell("validation", True), cell("459개 ID를 val.csv에서 읽어 직접 지정")], [cell("오프라인", True), cell("local weight만 사용, API/VLM 호출 없음")], [cell("정답", True), cell("YYYY-MM-DD; year missing은 NoNE")], [cell("selector", True), cell("ITDA train에서 fitting, validation에는 고정 weight 적용")]]
    story.append(styled_table(protocol, [35 * mm, 115 * mm], header=True))
    story.append(PageBreak())

    # Performance
    story.append(P("2. 정확도 및 속도 결과", "H1K"))
    story.append(P("모든 정확도는 동일한 459장 validation ID에 대해 다시 측정한 값입니다. YOLO-only와 full-image branch는 각각 fresh inference이고, union은 두 후보 집합을 train-fit spatial selector로 결합한 결과입니다.", "BodyK"))
    perf = [
        ["구조", "Final EM", "Candidate Recall", "Selection", "속도"],
        ["Full-image PP-OCRv5 + PP-OCRv6", pct(full["final_date_exact_match"]), pct(full["candidate_recall"]), pct(full["candidate_selection_accuracy"]), f'{full["sec_per_image"]:.3f} sec/img'],
        ["YOLO expiry crop only (960)", pct(yolo["final_em"]), pct(yolo["candidate_recall"]), "93.67%", f'{yolo["sec_per_image"]:.3f} sec/img'],
        ["Union + train-fit spatial selector", pct(union["final_em"]), pct(union["candidate_recall"]), pct(union["selection_accuracy"]), "~0.957 sec/img*"],
    ]
    story.append(styled_table(perf, [62 * mm, 22 * mm, 29 * mm, 25 * mm, 32 * mm]))
    story.append(P("* Union 속도는 두 branch를 순차 실행한 상한 추정치입니다. 조건부 fallback으로 실제 운영 시 더 짧아질 수 있습니다.", "SmallK"))
    story.append(Spacer(1, 4 * mm))
    story.append(P("구성요소 정확도", "H2K"))
    components = [["구조", "Year", "Month", "Day", "NONE 출력 비율"], ["Full-image PP-OCRv6", pct(full["year_accuracy"]), pct(full["month_accuracy"]), pct(full["day_accuracy"]), pct(full["none_rate"])]]
    story.append(styled_table(components, [62 * mm, 22 * mm, 22 * mm, 22 * mm, 42 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(P("주요 해석", "H2K"))
    bullets = [
        "YOLO crop 단독은 67.76% EM으로 전체 OCR보다 낮습니다.",
        "Union은 후보 재현율을 75.60%에서 85.19%로 끌어올립니다.",
        "spatial selector 적용 후 최종 EM은 80.61%가 됩니다.",
        "따라서 현재 가장 큰 개선 여지는 selection보다 recognition/candidate generation에 있습니다.",
    ]
    for item in bullets:
        story.append(P("- " + item, "BodyK"))
    story.append(PageBreak())

    # Bottleneck and environment
    story.append(P("3. 병목 분석", "H1K"))
    err = [["관찰 지표", "수치", "판정"], ["Full OCR Candidate Recall", pct(full["candidate_recall"]), "후보 생성에서 약 24.4% 누락"], ["Full OCR Selection", pct(full["candidate_selection_accuracy"]), "후보가 있으면 선택은 안정적"], ["Union Candidate Recall", pct(union["candidate_recall"]), "YOLO branch가 누락 후보를 보강"], ["Union Selection", pct(union["selection_accuracy"]), "공간 feature가 오선택을 제한"]]
    story.append(styled_table(err, [45 * mm, 35 * mm, 70 * mm]))
    story.append(Spacer(1, 5 * mm))
    story.append(P("Full-image 오류 분포 (잘못된 130장 기준)", "H2K"))
    errors = [["오류 유형", "Count", "오류 내부 비율", "의미"], ["DET", "1", "0.8%", "검출 영역 자체를 못 찾음"], ["GEN", "61", "46.9%", "문자열은 있으나 날짜 후보 생성 실패"], ["REC", "50", "38.5%", "날짜 문자 인식 오류"], ["SEL", "18", "13.8%", "후보 중 다른 날짜 선택"]]
    story.append(styled_table(errors, [25 * mm, 22 * mm, 35 * mm, 68 * mm]))
    story.append(Spacer(1, 5 * mm))
    story.append(P("결론: detector backbone 교체보다 날짜 crop 인식, 분리된 box line reconstruction, 숫자/도트매트릭스 강건화가 우선입니다.", "CalloutK"))
    story.append(P("환경 검증", "H2K"))
    env = [["항목", "검증 결과"], ["GPU accuracy cross-check", "Colab T4, Paddle 3.3.0 CUDA12.6, 459장"], ["CPU compatibility", "Paddle CPU 3.3.0 import 및 1장 inference 성공"], ["oneDNN", "enable_mkldnn=true는 PIR 오류; false로 정상 실행"], ["정확한 4-core latency", "현재 Colab 무료 CPU는 2 vCPU라 미측정 - 4 vCPU final gate 필요"], ["제출 스키마", "459 rows, image_id/year/month/day/final_date 검사 통과"]]
    story.append(styled_table(env, [45 * mm, 105 * mm]))
    story.append(PageBreak())

    # Next experiments and artifacts
    story.append(P("4. 다음 실험 우선순위", "H1K"))
    next_rows = [["순위", "실험", "목표", "판정 기준"], ["1", "tight + context dual crop", "날짜 문자와 expiry keyword를 동시에 보존", "Candidate Recall 상승"], ["2", "320x64 / 480x64 / perspective crop", "잘림, 기울기, 도트매트릭스 완화", "REC/GEN 감소"], ["3", "line reconstruction", "2027. / 10. / 14 분리 box 재조합", "GEN 감소"], ["4", "Kaggle + ITDA mixed fine-tuning", "crop domain overfit 방지", "validation EM 상승"], ["5", "conditional second pass", "저신뢰 샘플만 재인식", "정확도 유지 + CPU 시간 감소"], ["6", "date/due/code + relation selector", "semantic/spatial 오선택 감소", "Selection 1-2%p 개선"]]
    story.append(styled_table(next_rows, [15 * mm, 52 * mm, 55 * mm, 28 * mm]))
    story.append(Spacer(1, 5 * mm))
    story.append(P("주의할 점", "H2K"))
    story.append(P("이제 구조적 큰 폭 개선은 쉽지 않습니다. 현재 80.61%에서 82-85%는 인식 후보를 늘리는 실험으로 현실적인 목표지만, 90% 이상은 domain-specific recognizer 학습과 더 강한 box/문자 supervision이 필요합니다. 모든 후보는 test split을 열기 전에 validation에서만 비교해야 합니다.", "BodyK"))
    story.append(P("재현 산출물", "H2K"))
    artifacts = [
        "runs/colab/t4_full_yolo_union_trainfit_selector.json - union accuracy",
        "runs/colab/colab_full_baseline_v6_val459_correct_metrics.json - full OCR metrics",
        "runs/colab/t4_yolo_v6_accuracy_459.json - YOLO branch metrics",
        "docs/current_architecture_review_packet.md - 상세 아키텍처/감사 문서",
        "runs/colab/final_sota_competition_environment_report.md - 환경 검증 보고서",
    ]
    for item in artifacts:
        story.append(P("- " + item, "SmallK"))
    story.append(Spacer(1, 6 * mm))
    end = Table([[P("최종 판단", "CalloutWhiteK"), P("현재 최상위 모델은 full OCR + YOLO expiry union + train-fit spatial selector입니다. 다음 개선은 detector 교체가 아니라 recognition/candidate recall을 겨냥해야 합니다.", "BodyK")]], colWidths=[28 * mm, 122 * mm])
    end.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), NAVY), ("TEXTCOLOR", (0, 0), (0, 0), colors.white), ("BACKGROUND", (1, 0), (1, 0), PALE_BLUE), ("BOX", (0, 0), (-1, -1), 0.6, BLUE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    story.append(end)
    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    print(OUT)


if __name__ == "__main__":
    build()
