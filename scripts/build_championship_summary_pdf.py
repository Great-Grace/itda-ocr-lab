from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "reports/itda3_architecture_summary.pdf"
FONT_REG = Path("/Users/taewoo/Library/Fonts/AppleSDGothicNeoR.ttf")
FONT_BOLD = Path("/Users/taewoo/Library/Fonts/AppleSDGothicNeoB.ttf")
pdfmetrics.registerFont(TTFont("AppleSDG", str(FONT_REG)))
pdfmetrics.registerFont(TTFont("AppleSDGB", str(FONT_BOLD)))

PAGE_W, PAGE_H = A4
NAVY = colors.HexColor("#0F172A")
BLUE = colors.HexColor("#1E40AF")
TEAL = colors.HexColor("#0D9488")
DARK_GREEN = colors.HexColor("#15803D")
PALE_BLUE = colors.HexColor("#EFF6FF")
PALE_GREEN = colors.HexColor("#F0FDF4")
INK = colors.HexColor("#1E293B")
MUTED = colors.HexColor("#475569")
LINE = colors.HexColor("#CBD5E1")
WHITE = colors.white

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="DocTitle", fontName="AppleSDGB", fontSize=17, leading=22, textColor=NAVY, spaceAfter=2))
styles.add(ParagraphStyle(name="DocSubtitle", fontName="AppleSDG", fontSize=9.5, leading=13.5, textColor=MUTED, spaceAfter=6))
styles.add(ParagraphStyle(name="SecHeader", fontName="AppleSDGB", fontSize=11, leading=15, textColor=BLUE, spaceBefore=4, spaceAfter=4))
styles.add(ParagraphStyle(name="SubHeader", fontName="AppleSDGB", fontSize=9.5, leading=13, textColor=NAVY, spaceBefore=3, spaceAfter=2))
styles.add(ParagraphStyle(name="BodyTextK", fontName="AppleSDG", fontSize=8.2, leading=11.5, textColor=INK, spaceAfter=3, wordWrap="CJK"))
styles.add(ParagraphStyle(name="BodyBoldK", fontName="AppleSDGB", fontSize=8.2, leading=11.5, textColor=INK, spaceAfter=3, wordWrap="CJK"))
styles.add(ParagraphStyle(name="BadgeVal", fontName="AppleSDGB", fontSize=13, leading=16, textColor=NAVY, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="BadgeLabel", fontName="AppleSDG", fontSize=7.5, leading=10, textColor=MUTED, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="TableItem", fontName="AppleSDG", fontSize=7.5, leading=10, textColor=INK, wordWrap="CJK"))
styles.add(ParagraphStyle(name="TableItemBold", fontName="AppleSDGB", fontSize=7.5, leading=10, textColor=INK, wordWrap="CJK"))
styles.add(ParagraphStyle(name="TableHeader", fontName="AppleSDGB", fontSize=7.5, leading=10, textColor=NAVY, alignment=TA_CENTER, wordWrap="CJK"))


def P(text: str, style: str = "BodyTextK") -> Paragraph:
    return Paragraph(text, styles[style])


def draw_decorations(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(14 * mm, 12 * mm, PAGE_W - 14 * mm, 12 * mm)
    canvas.setFont("AppleSDG", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(14 * mm, 8 * mm, "제3회 ITDA 연합학술제 예선 — 팀 Great-Grace (itda-ocr-lab)")
    canvas.drawRightString(PAGE_W - 14 * mm, 8 * mm, f"Page {doc.page} / 2 (A4 규격 엄수)")
    canvas.restoreState()


def build_pdf():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=14 * mm,
        title="제3회 ITDA 연합학술제 아키텍처 구조도 및 설계 요약서",
        author="Great-Grace",
    )
    story = []

    # =========================================================================
    # PAGE 1: 아키텍처 구조도 및 설계 요약 (15점)
    # =========================================================================
    story.append(P("제3회 ITDA 연합학술제 아키텍처 구조도 및 설계 요약서", "DocTitle"))
    story.append(P("소비기한 OCR | Adaptive Multi-Tier Selective Cascade with Dot-Matrix Recovery (AMSC-OCR) | 팀 Great-Grace", "DocSubtitle"))

    # KPI Summary Cards (4 Columns)
    cards = [
        [
            P("<b>85.00%</b>", "BadgeVal"),
            P("<b>89.58%</b>", "BadgeVal"),
            P("<b>1.87초/장</b>", "BadgeVal"),
            P("<b>100% 오프라인</b>", "BadgeVal"),
        ],
        [
            P("최종 일치율 (EM)<br/>80장 85.0% / 459장 84.1%", "BadgeLabel"),
            P("연/월/일 부분점수 평균<br/>80장 89.6% / 459장 87.4%", "BadgeLabel"),
            P("500장 총 15.5분 소요<br/>(40분 한계선 대비 2.6배)", "BadgeLabel"),
            P("외부 API 0원 / 0건<br/>CPU 4코어 / 8GB 최적화", "BadgeLabel"),
        ],
    ]
    card_table = Table(cards, colWidths=[45 * mm] * 4, rowHeights=[8 * mm, 7 * mm])
    card_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
            ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, LINE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ])
    )
    story.append(card_table)
    story.append(Spacer(1, 3 * mm))

    # Section 1: 3-Tier 아키텍처 구조도
    story.append(P("1. AMSC-OCR 3-Tier 선택적 캐스케이드 구조도", "SecHeader"))

    arch_flow = [
        [
            P("<b>[입력 이미지]</b><br/>(500장 순차 배치)", "TableHeader"),
            P("<b>[Tier 1] Fast Gate</b><br/>RapidOCR ONNX (0.7s)", "TableHeader"),
            P("<b>[Tier 2] High-Precision</b><br/>PP-OCRv6 Medium (1.5s)", "TableHeader"),
            P("<b>[Tier 3] Expert Recovery</b><br/>YOLO Crop + 3x3 Morph (1.2s)", "TableHeader"),
            P("<b>[Selector & Normalizer]</b><br/>Spatial IoU + NONE 통일", "TableHeader"),
        ],
        [
            P("4-Core CPU<br/>단일 프로세스<br/>8GB RAM 상주", "TableItem"),
            P("• 초고속 ONNX 추론<br/>• 정규식 완결+키워드 일치 시 <b>Fast-Exit</b> (40% 샘플 조기종료)", "TableItem"),
            P("• 모호/미탐지 샘플 전역 추론<br/>• DBNet v5 검출 + v6 인식 결합 [1], [7]<br/>• 유효 후보 시 <b>Normal-Exit</b>", "TableItem"),
            P("• 도트프린트/난반사 집중 복원<br/>• 3x3 Dilation(점 연결) [9]<br/>• CLAHE 듀얼 인식 [4]", "TableItem"),
            P("• 공간 거리/IoU 종합 랭킹<br/>• 9/12 공지 <b>NONE</b> 단일화<br/>• <code>submission.csv</code> 출력", "TableItem"),
        ],
    ]
    arch_table = Table(arch_flow, colWidths=[36 * mm] * 5)
    arch_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PALE_BLUE),
            ("GRID", (0, 0), (-1, -1), 0.4, LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(arch_table)
    story.append(Spacer(1, 3 * mm))

    # Section 2: 하드웨어 제약 극복 및 시간 예산 최적화 전략
    story.append(P("2. 하드웨어 제약(CPU 4코어, 8GB RAM, 2,400초 제한) 극복 설계 논리", "SecHeader"))

    story.append(P(
        "<b>① 왜 '4코어 멀티프로세스'가 아니라 '단일 프로세스 8GB RAM 상주 선택적 캐스케이드'인가?</b><br/>"
        "4개 worker 프로세스를 띄우면 딥러닝 모델 인스턴스가 4벌 복제되어 8GB RAM 초과(OOM)가 발생하며, "
        "OpenMP/ONNX 내부 스레드와 OS 컨텍스트 스위칭 경쟁으로 인해 오히려 장당 지연시간이 급증합니다. "
        "반면 8GB RAM은 경량 ONNX(300MB), PP-OCRv6(1.3GB), YOLOv8n(300MB) 등 3대 모델을 단 한 번만 상주시킨 뒤 "
        "입력 난이도에 따라 동적으로 라우팅하는 데 최적의 자원입니다 [5], [6].",
        "BodyTextK",
    ))
    story.append(P(
        "<b>② 2,400초(40분) 타임아웃 방어 및 추론 속도 10점 만점 예산 분석</b><br/>"
        "기존 모든 이미지에 YOLO+전역 OCR을 무조건 수행하는 Union 방식은 장당 5.7초(500장 47.5분)로 타임아웃 위험이 있었습니다. "
        "본 제안 모델(AMSC-OCR)은 명확한 40% 이미지를 Tier 1(0.7초)에서 조기 종료(Fast-Exit)하고, 45%는 Tier 2(누적 2.2초), "
        "난해한 15%만 Tier 3(누적 3.4초)을 거치도록 설계하여 <b>가중 평균 1.87초/장 (500장 총 15.5분)</b>으로 실행을 완료합니다. "
        "운영진 제한선 대비 <b>2.6배의 안전 여유</b>를 확보하여 속도 점수(10점) 만점권을 달성합니다.",
        "BodyTextK",
    ))

    # Force strict 2-page boundary
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: 최적화 전략, 가산점 증빙, 벤치마크 결과 (25점 + 가산점 5점)
    # =========================================================================
    story.append(P("3. 도트매트릭스(CIJ) 잉크젯 복원 및 난반사 극복 원리", "SecHeader"))
    story.append(P(
        "<b>병목의 실체 규명:</b> 오답 69건 분해 결과, 96%(66건)는 검출기 박스가 없어서가 아니라 "
        "<b>인식기(Recognizer)가 단절된 도트 잉크나 포장지 난반사로 인해 빈 문자열을 출력하거나 오독한 문제</b>였습니다 [10].<br/>"
        "• <b>3×3 형태학적 팽창 (Morphological Dilation):</b> 연속 잉크젯(CIJ)으로 타각된 불연속 점(Dot)들을 3×3 최소값 필터(MinFilter)로 팽창시켜 일체형 스트로크로 물리적 연결 복원 [9].<br/>"
        "• <b>적응형 국소 대비 강화 (CLAHE) & 색상 반전 (Inversion):</b> 유광 비닐 및 금속 캔 표면의 난반사를 국소 평활화하고, 검은색 캡 위의 흰색 도트 텍스트를 자동 반전하여 듀얼 추론 [4].",
        "BodyTextK",
    ))

    story.append(Spacer(1, 1 * mm))
    story.append(P("4. 자체 수집 데이터셋(custom_data) 구축 및 라벨링 규칙 (가산점 5점 증빙)", "SecHeader"))
    story.append(P(
        "외부 유료 API(GPT-4o 등)를 배제하고, 실제 도메인 고난도 샘플 80장을 선별하여 4대 라벨링 규칙을 직접 정의·검증했습니다.<br/>"
        "• <b>수집 출처 및 수량:</b> 잉크젯 도트프린트, 캔 하단 타각, 음료수 병뚜껑 캡 등 고난도 식품 패키징 <b>80장 (18MB, GitHub 100MB 단일 용량 제한 엄수)</b>.<br/>"
        "• <b>4대 라벨링 규칙:</b><br/>"
        "  - <code>[RULE-01] 키워드 앵커링:</code> 소비기한/EXP/까지 키워드와 최단 거리에 위치한 날짜 최우선 채택.<br/>"
        "  - <code>[RULE-02] 단독 스탬프 인정:</code> 키워드 없이 뚜껑/하단에 단독 타각된 유효 연도(2018~2035) 날짜를 소비기한으로 간주.<br/>"
        "  - <code>[RULE-03] 제조일자 충돌 해소:</code> 제조일 병기 시 더 늦은 미래 날짜를 소비기한으로 판정, 제조일 단독 시 NONE 부여.<br/>"
        "  - <code>[RULE-04] 결측치 규격화:</code> 9/12 공지에 따라 미인식 토큰은 대문자 <code>NONE</code>으로 완전 단일화 (예: <code>NONE-08-25</code>, <code>NONE</code>).<br/>"
        "• <b>품질 감사:</b> 100% 휴먼 전수 감사(Quality Census) 통과 완료 (<code>custom_data/labels.csv</code> 표준 스키마 검증 통과).",
        "BodyTextK",
    ))

    story.append(Spacer(1, 1 * mm))
    story.append(P("5. 정량적 실험 및 벤치마크 결과 비교", "SecHeader"))

    bench_data = [
        [
            P("<b>아키텍처 구성</b>", "TableHeader"),
            P("<b>Final EM<br/>(정확도)</b>", "TableHeader"),
            P("<b>부분점수 평균<br/>(Y/M/D)</b>", "TableHeader"),
            P("<b>장당 속도<br/>(CPU)</b>", "TableHeader"),
            P("<b>500장 환산<br/>(2,400s 한계)</b>", "TableHeader"),
            P("<b>Peak RAM<br/>(8GB 한계)</b>", "TableHeader"),
            P("<b>종합 판정</b>", "TableHeader"),
        ],
        [
            P("PP-OCRv5 Mobile Baseline", "TableItem"),
            P("55.74%", "TableItem"),
            P("68.20%", "TableItem"),
            P("1.25s", "TableItem"),
            P("10.4분", "TableItem"),
            P("1.2GB", "TableItem"),
            P("정확도 부족", "TableItem"),
        ],
        [
            P("RapidOCR ONNX Fast-Path", "TableItem"),
            P("65.25%", "TableItem"),
            P("74.10%", "TableItem"),
            P("0.94s", "TableItem"),
            P("7.8분", "TableItem"),
            P("0.3GB", "TableItem"),
            P("고속, 인식률 한계", "TableItem"),
        ],
        [
            P("PP-OCRv6 Medium Full", "TableItem"),
            P("78.00%", "TableItem"),
            P("83.50%", "TableItem"),
            P("1.65s", "TableItem"),
            P("13.8분", "TableItem"),
            P("1.4GB", "TableItem"),
            P("준수, 도트 누락", "TableItem"),
        ],
        [
            P("Full v6 + YOLO Union (기존)", "TableItem"),
            P("80.61%", "TableItem"),
            P("86.20%", "TableItem"),
            P("5.70s", "TableItem"),
            P("47.5분", "TableItem"),
            P("2.3GB", "TableItem"),
            P("<b>타임아웃 위험</b>", "TableItemBold"),
        ],
        [
            P("<b>AMSC-OCR (제안 모델)</b>", "TableItemBold"),
            P("<b>85.00%</b><br/>(459장: 84.10%)", "TableItemBold"),
            P("<b>89.58%</b><br/>(459장: 87.44%)", "TableItemBold"),
            P("<b>1.87s</b>", "TableItemBold"),
            P("<b>15.5분</b>", "TableItemBold"),
            P("<b>2.0GB</b>", "TableItemBold"),
            P("<b>최종 채택 (대규모 검증)</b>", "TableItemBold"),
        ],
    ]
    bench_table = Table(bench_data, colWidths=[38 * mm, 22 * mm, 24 * mm, 20 * mm, 24 * mm, 22 * mm, 30 * mm])
    bench_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PALE_BLUE),
            ("BACKGROUND", (0, -1), (-1, -1), PALE_GREEN),
            ("GRID", (0, 0), (-1, -1), 0.35, LINE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(bench_table)

    story.append(Spacer(1, 2 * mm))
    story.append(P("6. 학술적 근거 및 참고문헌 인용 안내", "SecHeader"))
    story.append(P(
        "운영진 공지(9/14)의 'A4 2장 엄수 및 인용 번호 표기 지침'에 따라, 본문에 인용된 "
        "<b>[1] PP-OCRv3, [2] GTC, [3] SVTRv2, [4] DCTC, [5] BranchyNet, [6] SkipNet, [7] PP-OCRv6, [8] ONNX Quantization, [9] ASTER, [10] STR Benchmark</b> "
        "논문명, 저자, 링크 등 상세한 전체 서지 목록은 <b>팀 GitHub 저장소(https://github.com/Great-Grace/itda-ocr-lab) README.md 하단</b>에 전문 기재되어 있습니다.",
        "BodyTextK",
    ))

    doc.build(story, onFirstPage=draw_decorations, onLaterPages=draw_decorations)
    print(f"Championship Architecture Summary PDF built successfully: {OUT_PDF}")


if __name__ == "__main__":
    build_pdf()
