from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def generate_reports(
    run_dir: Path,
    predictions: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    labels_path: str | Path | None = None,
) -> tuple[Path, Path]:
    labels: dict[str, str] = {}
    if labels_path:
        import csv

        with Path(labels_path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                labels[str(row["image_id"])] = str(row["final_date"])

    # Attach ground truth & correctness to review rows for reporting
    enhanced_rows: list[dict[str, Any]] = []
    for r in review_rows:
        img_id = r["image_id"]
        gt = labels.get(img_id, "N/A")
        pred = r["final_date"]
        if gt != "N/A":
            is_correct = pred == gt
        else:
            is_correct = None
        row_copy = dict(r)
        row_copy["ground_truth"] = gt
        row_copy["is_correct"] = is_correct
        enhanced_rows.append(row_copy)

    summary_path = run_dir / "summary.md"
    html_path = run_dir / "review.html"

    _write_summary_md(summary_path, metrics, enhanced_rows)
    _write_review_html(html_path, metrics, enhanced_rows)

    return summary_path, html_path


def _write_summary_md(path: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# OCR Experiment Summary",
        "",
        "## Key Performance Indicators",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Total Images** | {metrics.get('image_count', len(rows))} |",
    ]
    if "labeled_count" in metrics:
        exact_match = metrics.get("final_date_exact_match", 0.0)
        lines.extend([
            f"| **Labeled Count** | {metrics['labeled_count']} |",
            f"| **Exact Match Accuracy** | **{exact_match * 100:.2f}%** ({exact_match:.4f}) |",
        ])
    lines.extend([
        f"| **Mean Latency** | {metrics.get('latency_ms_mean', 0.0):.2f} ms |",
        f"| **P95 Latency** | {metrics.get('latency_ms_p95', 0.0):.2f} ms |",
        f"| **NONE Prediction Rate** | {metrics.get('none_rate', 0.0) * 100:.2f}% |",
        "",
    ])

    # Highlight failures
    failures = [r for r in rows if r["is_correct"] is False]
    if failures:
        lines.append(f"## Failure Cases (Total {len(failures)})")
        lines.append("")
        lines.append("| Image ID | Ground Truth | Prediction | Candidate | OCR Text | Evidence |")
        lines.append("|---|---|---|---|---|---|")
        for f in failures[:10]:
            ocr_snippet = f['ocr_text'][:40] + ("..." if len(f['ocr_text']) > 40 else "")
            lines.append(
                f"| `{f['image_id']}` | `{f['ground_truth']}` | `{f['final_date']}` | `{f['candidate']}` | {ocr_snippet} | {f['evidence']} |"
            )
        if len(failures) > 10:
            lines.append(f"*(and {len(failures) - 10} more failures; see review.html or review.csv)*")
        lines.append("")
    elif any(r["is_correct"] is True for r in rows):
        lines.append("## Result: All labeled samples passed! 🎉\n")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_review_html(path: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    exact_match_str = f"{metrics.get('final_date_exact_match', 0.0) * 100:.1f}%" if "final_date_exact_match" in metrics else "N/A"
    # Prevent OCR text from terminating the script block in the standalone
    # viewer. Values are also escaped again before row-level HTML rendering.
    data_json = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>OCR Experiment Review</title>
<style>
  :root {{
    --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #f8fafc;
    --text-muted: #94a3b8; --accent: #38bdf8; --success: #22c55e; --danger: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); padding: 24px; }}
  .container {{ max-width: 1300px; margin: 0 auto; }}
  header {{ margin-bottom: 24px; }}
  h1 {{ font-size: 24px; margin-bottom: 8px; color: var(--accent); }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
  .card-title {{ font-size: 13px; color: var(--text-muted); margin-bottom: 6px; }}
  .card-val {{ font-size: 26px; font-weight: bold; }}
  .controls {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
  input[type="text"] {{ background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; color: var(--text); flex: 1; min-width: 240px; }}
  .btn-group button {{ background: var(--card); border: 1px solid var(--border); color: var(--text); padding: 8px 16px; cursor: pointer; border-radius: 6px; margin-right: 6px; }}
  .btn-group button.active {{ background: var(--accent); color: #000; font-weight: bold; border-color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; font-size: 14px; }}
  th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: #182234; color: var(--text-muted); font-size: 12px; text-transform: uppercase; }}
  tr:hover {{ background: #253349; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
  .badge-pass {{ background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid var(--success); }}
  .badge-fail {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }}
  .badge-none {{ background: rgba(148, 163, 184, 0.2); color: var(--text-muted); }}
  .ocr-preview {{ max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-muted); }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>ITDA OCR Lab — Review Dashboard</h1>
    <p style="color: var(--text-muted);">실험 상세 결과 및 오인식 분석 뷰어</p>
  </header>

  <div class="cards">
    <div class="card"><div class="card-title">정확도 (Exact Match)</div><div class="card-val" style="color: var(--success);">{exact_match_str}</div></div>
    <div class="card"><div class="card-title">전체 이미지 수</div><div class="card-val">{metrics.get('image_count', len(rows))}</div></div>
    <div class="card"><div class="card-title">평균 처리 시간</div><div class="card-val">{metrics.get('latency_ms_mean', 0.0):.1f} ms</div></div>
    <div class="card"><div class="card-title">NONE 비율</div><div class="card-val">{metrics.get('none_rate', 0.0) * 100:.1f}%</div></div>
  </div>

  <div class="controls">
    <input type="text" id="search" placeholder="이미지 ID, 날짜 또는 OCR 텍스트 검색..." oninput="render()">
    <div class="btn-group">
      <button class="active" onclick="setFilter('all', this)">전체</button>
      <button onclick="setFilter('fail', this)">오답 (Mismatch)</button>
      <button onclick="setFilter('pass', this)">정답 (Match)</button>
      <button onclick="setFilter('none', this)">NONE</button>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Image ID</th>
        <th>Ground Truth</th>
        <th>Prediction</th>
        <th>Candidate</th>
        <th>Confidence</th>
        <th>Latency</th>
        <th>OCR Text Snippet</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<script>
  const data = {data_json};
  let currentFilter = 'all';

  function escapeHtml(value) {{
    return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  }}

  function setFilter(f, btn) {{
    currentFilter = f;
    document.querySelectorAll('.btn-group button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    render();
  }}

  function render() {{
    const query = document.getElementById('search').value.toLowerCase();
    const tbody = document.getElementById('tbody');
    tbody.innerHTML = '';

    const filtered = data.filter(row => {{
      if (currentFilter === 'fail' && row.is_correct !== false) return false;
      if (currentFilter === 'pass' && row.is_correct !== true) return false;
      if (currentFilter === 'none' && row.final_date !== 'NONE') return false;
      if (query) {{
        const target = (row.image_id + ' ' + (row.ground_truth || '') + ' ' + row.final_date + ' ' + (row.ocr_text || '')).toLowerCase();
        if (!target.includes(query)) return false;
      }}
      return true;
    }});

    filtered.forEach(row => {{
      const tr = document.createElement('tr');
      let badge = '<span class="badge badge-none">N/A</span>';
      if (row.is_correct === true) badge = '<span class="badge badge-pass">PASS</span>';
      else if (row.is_correct === false) badge = '<span class="badge badge-fail">FAIL</span>';

      tr.innerHTML = `
        <td><strong>${{escapeHtml(row.image_id)}}</strong></td>
        <td>${{escapeHtml(row.ground_truth || '-')}}</td>
        <td>${{badge}} <code>${{escapeHtml(row.final_date)}}</code></td>
        <td>${{escapeHtml(row.candidate)}}</td>
        <td>${{(row.confidence || 0).toFixed(2)}}</td>
        <td>${{row.elapsed_ms}} ms</td>
        <td class="ocr-preview" title="${{escapeHtml(row.ocr_text || '')}}">${{escapeHtml(row.ocr_text || '-')}}</td>
      `;
      tbody.appendChild(tr);
    }});
  }}

  render();
</script>
</body>
</html>
"""
    path.write_text(html_content, encoding="utf-8")
