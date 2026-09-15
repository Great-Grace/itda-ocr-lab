"""Build a range-readable manifest for the public Kaggle expiry OCR archive.

The archive is about 2.7 GB. This command downloads only the ZIP central
directory (a few MB) and records offsets so individual images/labels can be
fetched later with HTTP Range requests. It never materializes the archive.
"""
from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path


def _open_archive_url(api_url: str) -> tuple[str, int]:
    response = urllib.request.urlopen(urllib.request.Request(api_url), timeout=60)
    return response.geturl(), int(response.headers["Content-Length"])


def _zip64_values(extra: bytes, usize: int, csize: int, offset: int) -> tuple[int, int, int]:
    if not any(v == 0xFFFFFFFF for v in (usize, csize, offset)):
        return usize, csize, offset
    p = 0
    while p + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, p)
        body = extra[p + 4 : p + 4 + field_size]
        p += 4 + field_size
        if field_id != 0x0001:
            continue
        q = 0
        if usize == 0xFFFFFFFF:
            usize = struct.unpack_from("<Q", body, q)[0]; q += 8
        if csize == 0xFFFFFFFF:
            csize = struct.unpack_from("<Q", body, q)[0]; q += 8
        if offset == 0xFFFFFFFF:
            offset = struct.unpack_from("<Q", body, q)[0]
        break
    return usize, csize, offset


def _parse_central_directory(blob: bytes) -> list[dict]:
    rows: list[dict] = []
    p = 0
    while p + 46 <= len(blob) and blob[p : p + 4] == b"PK\x01\x02":
        fields = struct.unpack_from("<4s6H3L5H2L", blob, p)
        _, version_made, version_needed, flags, method, mtime, mdate, crc, csize, usize, name_len, extra_len, comment_len, disk, int_attr, ext_attr, offset = fields
        name_start = p + 46
        name = blob[name_start : name_start + name_len].decode("utf-8", "replace")
        extra = blob[name_start + name_len : name_start + name_len + extra_len]
        usize, csize, offset = _zip64_values(extra, usize, csize, offset)
        rows.append({
            "name": name,
            "crc": str(crc),
            "compressed_size": int(csize),
            "uncompressed_size": int(usize),
            "offset": int(offset),
            "compression": int(method),
            "flags": int(flags),
        })
        p = name_start + name_len + extra_len + comment_len
    if p != len(blob):
        raise ValueError(f"central directory parse stopped at {p}/{len(blob)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="scratch/kaggle_manifest.json")
    parser.add_argument("--dataset", default="kimhyeongminkhu/korean-expiry-date-ocr")
    args = parser.parse_args()

    api_url = f"https://www.kaggle.com/api/v1/datasets/download/{args.dataset}"
    archive_url, total_size = _open_archive_url(api_url)
    tail_size = min(total_size, 8_000_000)
    start = total_size - tail_size
    request = urllib.request.Request(archive_url, headers={"Range": f"bytes={start}-{total_size - 1}"})
    tail = urllib.request.urlopen(request, timeout=120).read()
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    _, _, _, _, _, central_size, central_offset, comment_len = struct.unpack_from("<4s4H2LH", tail, eocd)
    if central_offset < start:
        raise RuntimeError("central directory is larger than the downloaded tail")
    central_start = central_offset - start
    central = tail[central_start : central_start + central_size]
    rows = _parse_central_directory(central)
    payload = {"archive_url": archive_url, "archive_size": total_size, "files": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(rows), "archive_size": total_size, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
