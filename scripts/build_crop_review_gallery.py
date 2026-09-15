"""Create a local, exportable review gallery for date-line crop validation."""
from __future__ import annotations

import argparse
import csv
import html
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.crops), Path(args.output)
    assets = output / "crops"; assets.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((source / "labels.csv").open(encoding="utf-8")))
    cards = []
    for row in rows:
        image = source / row["image"]
        shutil.copy2(image, assets / row["image"])
        image_id = html.escape(row["image_id"]); label = html.escape(row["label"]); raw = html.escape(row.get("raw_text", ""))
        cards.append(f'''<article class="card" data-id="{image_id}" data-label="{label}">
<header><strong>{image_id}</strong><span>GT: {label}</span></header>
<img src="crops/{html.escape(row['image'])}" alt="{image_id} crop">
<p>OCR 원문: <code>{raw}</code></p>
<label>판정 <select><option value="unreviewed">미확인</option><option value="accept">정답 한 줄</option><option value="multiple_lines">여러 줄 섞임</option><option value="wrong_crop">다른 날짜/영역</option><option value="illegible">판독 불가</option></select></label>
</article>''')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Date crop review</title>
<style>body{{font:14px system-ui;margin:20px;background:#f5f6f8;color:#171717}}header{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}button{{padding:8px 12px}}#grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin-top:16px}}.card{{background:white;border:1px solid #d7dbe0;border-radius:8px;padding:10px}}.card header{{justify-content:space-between}}.card img{{width:100%;height:78px;object-fit:contain;background:#eee;margin:8px 0}}code{{word-break:break-all}}select{{margin-left:6px}}</style>
<header><h1>Date-line crop review</h1><span>{len(rows)}개. 정답 날짜를 다시 쓰지 말고 crop 품질만 선택하세요.</span><button onclick="download()">선택 결과 CSV 저장</button></header><main id="grid">{''.join(cards)}</main>
<script>function download(){{let r=['image_id,final_date,decision'];document.querySelectorAll('.card').forEach(c=>r.push([c.dataset.id,c.dataset.label,c.querySelector('select').value].join(',')));let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([r.join('\\n')],{{type:'text/csv'}}));a.download='date_crop_review.csv';a.click();}}</script></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")
    print(f"{output / 'index.html'}")


if __name__ == "__main__":
    main()
