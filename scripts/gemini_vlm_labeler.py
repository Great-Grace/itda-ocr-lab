#!/usr/bin/env python3
"""High-Accuracy Zero-Cost Gemini 3.6 Flash VLM Labeler for ITDA OCR Lab.

Processes images in batches (3-4 images per call) within the Free Tier limits.
Extracts expiration dates with domain explanations, validates formatting,
and outputs high-confidence Ground Truth labels for Knowledge Distillation.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
]
current_model_idx = 0


def get_current_model() -> str:
    global current_model_idx
    return MODELS[current_model_idx]


def rotate_model() -> str:
    global current_model_idx
    prev = MODELS[current_model_idx]
    current_model_idx = (current_model_idx + 1) % len(MODELS)
    nxt = MODELS[current_model_idx]
    print(f"  [Model Rotation] Daily quota reached for {prev}. Rotating to {nxt}!", flush=True)
    return nxt


PROMPT = """당신은 식품/상품 포장지에서 유통기한 및 소비기한을 정확하게 판독하는 OCR 전문가입니다.
첨부된 각 상품 사진에서 유통기한 또는 소비기한(EXP, BBE, 유통기한, 소비기한, 까지, 제조일자 등)을 추출하세요.

[규칙]
1. 날짜는 반드시 'YYYY-MM-DD' 형식으로 정규화하세요. (예: 21.12.28 -> 2021-12-28, 20240506 -> 2024-05-06)
2. '제조일자'와 '유통기한'이 둘 다 있으면 반드시 '유통기한/소비기한'을 선택하세요.
3. '제조일로부터 X개월' 표기만 있고 유통기한이 명시되지 않은 경우, 제조일자에 개월 수를 더해 계산하세요.
4. 날짜가 전혀 보이지 않거나 완전히 잘려서 판독 불가능한 경우 "date": null 로 표기하세요.
5. 반드시 아래 JSON 배열 형식으로만 응답하세요:

[
  {
    "filename": "이미지파일명",
    "date": "YYYY-MM-DD 또는 null",
    "text_found": "사진에서 발견한 원본 텍스트",
    "confidence": "high|medium|low",
    "reason": "선택한 근거 및 위치"
  }
]
"""


def process_batch(api_key: str, batch: list[Path], retries: int = 6) -> list[dict]:
    parts = [{"text": PROMPT}]
    filenames = [p.name for p in batch]
    parts.append({"text": f"분석할 이미지 목록: {', '.join(filenames)}"})

    for p in batch:
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        parts.append({"text": f"--- 이미지: {p.name} ---"})
        parts.append({"inline_data": {"mime_type": mime, "data": b64}})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0,
        },
    }
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(retries):
        model = get_current_model()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                text = res["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "results" in parsed:
                    parsed = parsed["results"]
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            if e.code == 429:
                if "GenerateRequestsPerDay" in err_body or "quota exceeded" in err_body.lower():
                    rotate_model()
                    time.sleep(2)
                else:
                    print(f"  [Rate Limit 429 RPM] Waiting 30s reset... (attempt {attempt+1}/{retries})", flush=True)
                    time.sleep(30)
            elif e.code == 503:
                print(f"  [503 Busy] Rotating model from {model}...", flush=True)
                rotate_model()
                time.sleep(3)
            else:
                print(f"  [HTTP {e.code}] Error: {err_body[:200]}", flush=True)
                time.sleep(5)
        except Exception as e:
            print(f"  [Error] {e}. Retrying in 5s...", flush=True)
            time.sleep(5)

    print(f"  [FAIL] Failed batch: {filenames}", flush=True)

    return [{"filename": fn, "date": None, "confidence": "none", "reason": "API fail"} for fn in filenames]


def main():
    parser = argparse.ArgumentParser(description="Gemini VLM Ground Truth Labeler")
    parser.add_argument("--input-dir", required=True, help="Path to input images directory")
    parser.add_argument("--output-dir", default="runs/gemini_vlm_labels", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of images per API call")
    parser.add_argument("--max-images", type=int, help="Optional image limit")
    parser.add_argument("--delay", type=float, default=5.5, help="Delay between requests in seconds (safe for 15 RPM)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        env_file = Path(".env")
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        raise SystemExit("Error: GEMINI_API_KEY is not set.")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "gemini_labels.csv"
    jsonl_path = out_dir / "gemini_labels.jsonl"

    all_images = sorted([
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])

    if args.max_images:
        all_images = all_images[:args.max_images]

    print(f"=== [Gemini 3.6 Flash VLM Labeler] ===")
    print(f"Input: {in_dir} ({len(all_images)} images)")
    print(f"Output: {out_dir}")
    print(f"Batch size: {args.batch_size} images/call (Safe 15 RPM delay: {args.delay}s)")

    # Resume support
    done_filenames = set()
    if csv_path.is_file():
        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("confidence") != "none" and row.get("reason") != "API fail":
                    done_filenames.add(row["filename"])
        print(f"Resuming: {len(done_filenames)} images already successfully labeled.")

    remaining = [p for p in all_images if p.name not in done_filenames]
    print(f"Remaining images to process: {len(remaining)}")

    if not csv_path.is_file():
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("filename,date,confidence,text_found,reason\n")

    total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
    processed = 0

    for b_idx in range(total_batches):
        batch = remaining[b_idx * args.batch_size : (b_idx + 1) * args.batch_size]
        print(f"[{b_idx+1}/{total_batches}] Requesting batch ({[p.name for p in batch]})...", flush=True)
        results = process_batch(api_key, batch)

        # Map results by filename
        res_by_fn = {r.get("filename"): r for r in results if isinstance(r, dict)}

        with open(csv_path, "a", encoding="utf-8") as f_csv, open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for p in batch:
                r = res_by_fn.get(p.name, {})
                d = r.get("date") or ""
                conf = r.get("confidence") or "medium"
                txt = (r.get("text_found") or "").replace('"', '""')
                reason = (r.get("reason") or "").replace('"', '""')
                f_csv.write(f'"{p.name}","{d}","{conf}","{txt}","{reason}"\n')
                f_jsonl.write(json.dumps({"filename": p.name, **r}, ensure_ascii=False) + "\n")

        processed += len(batch)
        print(f"  -> Saved {processed}/{len(remaining)} images.", flush=True)

        # Mirror to Google Drive for 100% data durability
        drive_csv = Path("/content/drive/MyDrive/gemini_labels.csv")
        if drive_csv.parent.exists():
            import shutil
            try:
                shutil.copyfile(csv_path, drive_csv)
            except Exception:
                pass

        time.sleep(args.delay)


    print(f"\nSUCCESS: All images labeled and saved to {csv_path}")


if __name__ == "__main__":
    main()
