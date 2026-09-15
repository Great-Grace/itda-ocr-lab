"""Build a self-contained, evidence-backed final ITDA OCR HTML report."""
from __future__ import annotations

import csv
import html
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def num(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = list(csv.DictReader((ROOT / "runs/benchmark_all.csv").open(encoding="utf-8", newline="")))
    fresh = [row for row in rows if row.get("run_kind") == "fresh-ocr" and row.get("rank_eligible") == "True"]
    cached = [row for row in rows if row.get("run_kind") == "cached-ablation"]
    fresh.sort(key=lambda row: float(row.get("final_em") or 0), reverse=True)
    cached.sort(key=lambda row: float(row.get("final_em") or 0), reverse=True)
    return fresh, cached


def benchmark_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{esc(row.get('experiment'))}</td>"
            f"<td>{esc(row.get('test_set'))}</td>"
            f"<td>{esc(row.get('images'))}</td>"
            f"<td>{esc(row.get('run_kind'))}</td>"
            f"<td class=metric>{pct(row.get('final_em'))}</td>"
            f"<td class=metric>{pct(row.get('candidate_recall'))}</td>"
            f"<td class=metric>{pct(row.get('selection_acc'))}</td>"
            f"<td>{num(row.get('sec_per_image'))}</td>"
            f"<td>{num(row.get('peak_ram_mb'), 0)}</td>"
            f"<td><a href='../{esc(row.get('run'))}'>{esc(row.get('run'))}</a></td>"
            "</tr>"
        )
    return "\n".join(body)


def main() -> int:
    fresh, cached = load_rows()
    final_fit = json.loads((ROOT / "runs/vessl/cpu_final_val/fresh_spatial_selector_fit.json").read_text())
    gpu_baseline = json.loads((ROOT / "runs/vessl/cpu_final_val/gpu_fresh_baseline_metrics.json").read_text())
    gpu_detector = json.loads((ROOT / "runs/vessl/cpu_final_val/gpu_fresh_detector_eval.json").read_text())
    cpu_detector = json.loads((ROOT / "runs/vessl/cpu_final_val/cpu_detector_branch_val.json").read_text())
    html_doc = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ITDA OCR Lab — Final SOTA Architecture Report</title>
<style>
:root{{--ink:#17212b;--muted:#667482;--line:#dfe7ec;--paper:#f5f8fa;--card:#fff;--blue:#1c6e8c;--teal:#0f8b8d;--green:#14804a;--amber:#b56b00;--red:#a12a2a;--shadow:0 12px 30px rgba(23,33,43,.07)}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.55}} main{{max-width:1240px;margin:0 auto;padding:48px 28px 80px}} h1{{font-size:42px;line-height:1.1;margin:12px 0}} h2{{font-size:25px;margin:42px 0 14px;border-bottom:1px solid var(--line);padding-bottom:9px}} h3{{font-size:18px;margin:22px 0 8px}} p{{margin:8px 0 14px}} .eyebrow{{color:var(--blue);font-weight:800;letter-spacing:.11em;text-transform:uppercase;font-size:12px}} .lede{{font-size:19px;color:#3d4d5b;max-width:900px}} .hero{{background:linear-gradient(135deg,#e8f5f7,#fff);border:1px solid #cce6ea;border-radius:24px;padding:34px;box-shadow:var(--shadow)}} .verdict{{display:inline-flex;align-items:center;gap:9px;border-radius:999px;padding:8px 14px;background:#e3f5eb;color:var(--green);font-weight:800;margin:12px 0}} .verdict::before{{content:"";width:9px;height:9px;border-radius:50%;background:var(--green)}} .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}} .card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:var(--shadow)}} .card .label{{font-size:12px;color:var(--muted);font-weight:700}} .card .value{{font-size:30px;font-weight:850;margin-top:2px}} .card .note{{font-size:12px;color:var(--muted)}} .flow{{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;margin:18px 0}} .node{{background:#fff;border:1px solid #cfe0e7;border-radius:12px;padding:14px 16px;min-width:180px}} .node strong{{display:block;color:var(--blue)}} .arrow{{display:grid;place-items:center;color:var(--teal);font-size:24px}} .callout{{border-left:5px solid var(--blue);background:#eef7f9;padding:15px 18px;border-radius:8px;margin:14px 0}} .warn{{border-left-color:var(--amber);background:#fff7e8}} .danger{{border-left-color:var(--red);background:#fff0f0}} table{{width:100%;border-collapse:separate;border-spacing:0;background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow);font-size:13px}} th,td{{padding:10px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{background:#edf3f6;color:#40515e;font-size:12px;white-space:nowrap}} tr:last-child td{{border-bottom:0}} td.metric{{font-weight:800;white-space:nowrap}} td a{{color:var(--blue);text-decoration:none;word-break:break-word}} .scroll{{overflow:auto;border-radius:14px}} .bar-chart{{display:grid;gap:12px;margin:16px 0;max-width:900px}} .bar-row{{display:grid;grid-template-columns:250px 1fr 70px;gap:10px;align-items:center;font-size:13px}} .bar{{height:17px;background:#e6edf0;border-radius:999px;overflow:hidden}} .bar span{{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--teal));border-radius:inherit}} .status{{font-weight:800}} .ok{{color:var(--green)}} .partial{{color:var(--amber)}} .no{{color:var(--red)}} .small{{font-size:12px;color:var(--muted)}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .pill{{display:inline-block;padding:3px 8px;border-radius:999px;background:#edf3f6;color:#50606c;font-size:12px;margin:2px}} input{{width:100%;padding:12px 14px;border-radius:10px;border:1px solid var(--line);font:inherit;margin:10px 0 12px}} details{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:13px 16px;margin:12px 0}} summary{{font-weight:800;cursor:pointer}} footer{{margin-top:48px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:18px}} @media(max-width:850px){{.grid{{grid-template-columns:repeat(2,1fr)}}.two{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:150px 1fr 60px}} h1{{font-size:32px}}}} @media(max-width:520px){{main{{padding:28px 15px 60px}}.grid{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:115px 1fr 54px;font-size:11px}}}}
</style></head><body><main>
<section class="hero"><div class="eyebrow">ITDA OCR LAB · FINAL EVIDENCE REPORT</div><h1>80%를 넘긴 최종 구조와, 왜 작동했는가</h1><div class="verdict">내부 대회 검증 기준: 80.39% EM 달성</div><p class="lede">Kaggle의 동일 원본 box 라벨, PP-OCRv6 전체 OCR, expiry 후보 detector, 그리고 bbox 공간정보 selector를 결합한 결과입니다. 이 수치는 대회 내부 validation에서의 최고 실측치이며, 외부 학술 SOTA라고 과장하지 않습니다.</p><p class="small">작성일 {date.today().isoformat()} · 평가 population: ITDA train 2,145 / val 459 / test 459 · test는 미사용</p></section>
<div class="grid"><div class="card"><div class="label">최종 fresh val EM</div><div class="value">{pct(final_fit['val']['final_em'])}</div><div class="note">80.39% · 459 images</div></div><div class="card"><div class="label">Candidate Recall</div><div class="value">{pct(final_fit['val']['candidate_recall'])}</div><div class="note">정답 후보 존재 비율</div></div><div class="card"><div class="label">Selection Accuracy</div><div class="value">{pct(final_fit['val']['selection_accuracy'])}</div><div class="note">후보가 있을 때</div></div><div class="card"><div class="label">CPU 예상 union 속도</div><div class="value">5.73s</div><div class="note">full OCR 4.87 + detector branch 0.86</div></div></div>
<h2>Executive summary</h2><div class="two"><div><p><strong>가장 큰 발견:</strong> detector를 OCR로 대체한 것이 아니라, 기존 전체 OCR 후보와 Kaggle box 기반 후보를 합친 것이 효과적이었습니다.</p><p><strong>80% 돌파 요인:</strong> parser의 영문 월/한국어 날짜 오류를 고치고, OCR token box와 detector box의 IoU를 selector에 추가했습니다.</p></div><div class="callout warn"><strong>SOTA 판정:</strong> 이 결과는 ITDA 내부 validation에서의 강한 task-specific 최고점입니다. 동일 조건의 외부 학술 benchmark 비교 없이 “세계 SOTA”라고 부를 수는 없습니다. 다만 데이터 부족이 아닌 후보 생성·공간 선택을 해결한 재현 가능한 발견입니다.</div></div>
<h2>최종 아키텍처</h2><div class="flow"><div class="node"><strong>입력</strong>원본 상품 이미지<br><span class="small">전처리 없음</span></div><div class="arrow">→</div><div class="node"><strong>Branch A</strong>PP-OCRv5 mobile detector<br>PP-OCRv6 medium recognizer</div><div class="arrow">+</div><div class="node"><strong>Branch B</strong>YOLOv8n expiry binary<br><span class="small">date + due box 통합, 1280px</span></div><div class="arrow">→</div><div class="node"><strong>선택</strong>confidence + bbox IoU<br>train-fit spatial selector</div><div class="arrow">→</div><div class="node"><strong>출력</strong>YYYY-MM-DD<br>연도 없음은 NoNE</div></div>
<h2>성능 변화</h2><div class="bar-chart"><div class="bar-row"><span>PP-OCRv6 full-image fresh</span><div class="bar"><span style="width:{gpu_baseline['final_date_exact_match']*100:.2f}%"></span></div><b>{pct(gpu_baseline['final_date_exact_match'])}</b></div><div class="bar-row"><span>expiry detector branch</span><div class="bar"><span style="width:{gpu_detector['final_em']*100:.2f}%"></span></div><b>{pct(gpu_detector['final_em'])}</b></div><div class="bar-row"><span>union + fixed selector</span><div class="bar"><span style="width:78.43%"></span></div><b>78.43%</b></div><div class="bar-row"><span>train-fit spatial selector</span><div class="bar"><span style="width:{final_fit['val']['final_em']*100:.2f}%"></span></div><b>{pct(final_fit['val']['final_em'])}</b></div></div>
<div class="callout"><strong>수치 해석:</strong> fresh full OCR은 {pct(gpu_baseline['final_date_exact_match'])}였지만, union은 Candidate Recall을 {pct(final_fit['val']['candidate_recall'])}까지 끌어올렸고, 공간 selector가 선택 정확도를 {pct(final_fit['val']['selection_accuracy'])}로 회복해 최종 {pct(final_fit['val']['final_em'])}에 도달했습니다.</div>
<h2>전체 실측 벤치마크</h2><p>아래 표는 저장된 43개 fresh OCR 측정치를 모두 포함합니다. cached selector 실험은 OCR을 재실행하지 않았으므로 별도 표로 분리했습니다.</p><input id="filter" placeholder="실험명·split·구성 검색"><div class="scroll"><table id="fresh"><thead><tr><th>Experiment</th><th>Split</th><th>N</th><th>Kind</th><th>Final EM</th><th>Candidate Recall</th><th>Selection</th><th>sec/img</th><th>Peak RAM MB</th><th>Evidence</th></tr></thead><tbody>{benchmark_table(fresh)}</tbody></table></div>
<details><summary>Cached / parser / selector ablation 전체 {len(cached)}개</summary><p class="small">이 결과들은 후보 규칙 연구에는 유용하지만 recognizer·detector·latency architecture 순위로 사용할 수 없습니다. 전체 원본은 <a href="./benchmark_all.csv">benchmark_all.csv</a>에 있습니다.</p><div class="scroll"><table><thead><tr><th>Experiment</th><th>Split</th><th>N</th><th>Final EM</th><th>Recall</th><th>Selection</th><th>Run</th></tr></thead><tbody>{''.join(f"<tr><td>{esc(r.get('experiment'))}</td><td>{esc(r.get('test_set'))}</td><td>{esc(r.get('images'))}</td><td class=metric>{pct(r.get('final_em'))}</td><td class=metric>{pct(r.get('candidate_recall'))}</td><td class=metric>{pct(r.get('selection_acc'))}</td><td>{esc(r.get('run'))}</td></tr>" for r in cached)}</tbody></table></div></details>
<h2>마지막 Kaggle detector sweep</h2><div class="scroll"><table><thead><tr><th>Detector</th><th>Input</th><th>mAP50</th><th>mAP50-95</th><th>Union Final EM</th><th>판정</th></tr></thead><tbody><tr><td>expiry binary YOLOv8n</td><td>1280px</td><td>95.72%</td><td>67.01%</td><td><strong>77.996% → fresh 80.39%</strong></td><td class="ok">최종 채택</td></tr><tr><td>expiry/full YOLO11n</td><td>1280px</td><td>96.07%</td><td>66.04%</td><td>77.12%</td><td>mAP 1위, EM 2위</td></tr><tr><td>date/due/full YOLO11n</td><td>1280px</td><td>94.59%</td><td>64.39%</td><td>76.69%</td><td>보조 비교군</td></tr></tbody></table></div><p class="small">Detector mAP가 가장 높은 모델과 최종 날짜 EM이 가장 높은 모델은 달랐습니다. 대회 목표에는 candidate recall·selection·EM을 우선했습니다.</p>
<h2>병목 분석</h2><div class="two"><div class="card"><h3>Candidate miss 69장</h3><table><tbody><tr><td>detector box 없음</td><td class=metric>3</td></tr><tr><td>box는 있으나 날짜 문자열 없음</td><td class=metric>30</td></tr><tr><td>다른 날짜만 인식</td><td class=metric>36</td></tr></tbody></table><p class="small">후보 miss의 66/69(96%)가 detector가 아니라 crop recognizer/box 의미 불일치였습니다.</p></div><div class="card"><h3>Selection error 32장</h3><table><tbody><tr><td>baseline 후보가 잘못 선택</td><td class=metric>27</td></tr><tr><td>detector 후보가 잘못 선택</td><td class=metric>5</td></tr><tr><td>spatial IoU selector 후 selection</td><td class=metric>94.37%</td></tr></tbody></table><p class="small">bbox IoU를 넣자 후보 존재 조건 selection이 91.79%에서 94.37%로 상승했습니다.</p></div></div>
<h2>무효화·제외된 결과</h2><div class="callout danger"><strong>53개 조합 master leaderboard는 사용하지 않았습니다.</strong> 단일 token cache 재사용, hard-coded 보정값, 실제와 다른 latency/model size가 포함된 시뮬레이션 리더보드로 감사 결과 무효화되었습니다. <a href="./multi_architecture_leaderboard/audit_report.md">감사 보고서</a>와 <a href="./multi_architecture_leaderboard/master_leaderboard.md">무효화 공지</a>를 확인하십시오.</div><p>또한 random-init SVTR/CRNN/Attention과 clean crop PP-OCRv5 fine-tuning은 clean-crop 내부 점수와 별개로 실제 detector branch EM을 올리지 못해 최종 구조에서 제외했습니다.</p>
<h2>대회 환경 적합성 감사</h2><div class="scroll"><table><thead><tr><th>항목</th><th>검증 결과</th><th>상태</th><th>비고</th></tr></thead><tbody>
<tr><td>학습 장치</td><td>A100 SXM 80GB ×2</td><td class=ok>PASS</td><td>detector sweep·recognizer 학습에 사용</td></tr>
<tr><td>최종 inference 장치</td><td>CPU 4-core 고정</td><td class=partial>PARTIAL</td><td>CPU latency는 검증. fresh union 정확도는 GPU로 확인(사용자 승인)</td></tr>
<tr><td>CPU thread 제어</td><td>taskset 0–3, OMP/MKL/OPENBLAS=4</td><td class=ok>PASS</td><td>oversubscription 방지</td></tr>
<tr><td>CPU latency</td><td>full OCR 4.87s/img, detector branch {num(cpu_detector['sec_per_image'])}s/img</td><td class=ok>PASS</td><td>union 단순 합산 약 5.73s/img, 3,352장 약 5.3시간</td></tr>
<tr><td>RAM</td><td>fresh baseline peak 약 1.91GB</td><td class=ok>PASS</td><td>RAM 비공개 조건에서 보수적으로 기록</td></tr>
<tr><td>외부 API/VLM</td><td>inference 호출 없음</td><td class=ok>PASS</td><td>모델 weight는 사전 materialize</td></tr>
<tr><td>데이터 누수</td><td>train/val/test filename-disjoint</td><td class=ok>PASS</td><td>test는 아직 미사용</td></tr>
<tr><td>최종 CPU 정확도</td><td>459장 full union fresh exact 미실행</td><td class=partial>OPEN</td><td>CPU 37분 이상 소요되어 latency만 실측. 최종 제출 전 1회 권장</td></tr>
</tbody></table></div>
<h2>SOTA 판정과 다음 결정</h2><div class="two"><div><p><span class="pill">내부 대회 기준</span> 80.39%는 현재 확보된 실측 중 최고입니다. Kaggle box supervision을 실제 후보 recall 개선으로 연결한 점은 충분히 강한 task-specific 발견입니다.</p><p><span class="pill">학술 SOTA 기준</span> 아직 외부 표준 benchmark, 동일 split, 동일 metric 비교가 없어 학술적 SOTA 주장은 보류합니다.</p></div><div class="callout"><strong>남은 단 한 단계:</strong> selector 설정을 동결하고, CPU-only full union을 한 번 측정한 뒤 잠금 test를 실행하십시오. 그 전에는 80.39%를 최종 대회 점수로 부르지 않습니다.</div></div>
<h2>재현 자료</h2><p><a href="./vessl/final_sweep/final_sweep_report.md">최종 sweep 보고서</a> · <a href="../docs/current_architecture_review_packet.md">AI 리뷰용 상세 아키텍처 문서</a> · <a href="./vessl/final_sweep/selector_fit.json">train-fit selector</a> · <a href="./vessl/cpu_final_val/fresh_spatial_selector_fit.json">fresh spatial selector 결과</a> · <a href="../docs/kaggle_dataset_utilization.md">Kaggle 데이터 검증</a> · <a href="../docs/eighty_percent_bottleneck_analysis.md">80% 병목 분석</a> · <a href="./benchmark_review.md">기존 종합 benchmark review</a></p>
<footer>이 보고서는 저장된 실측 artifact를 바탕으로 생성되었습니다. Cached/invalidated/simulated 결과는 fresh end-to-end 결과와 구분했습니다. Test split은 보고서 작성 시점까지 사용하지 않았습니다.</footer>
</main><script>const filter=document.getElementById('filter');filter.addEventListener('input',()=>{{const q=filter.value.toLowerCase();document.querySelectorAll('#fresh tbody tr').forEach(r=>r.style.display=r.innerText.toLowerCase().includes(q)?'':'none')}});</script></body></html>'''
    output = ROOT / "runs/final_sota_report.html"
    output.write_text(html_doc, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
