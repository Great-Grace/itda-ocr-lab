#!/usr/bin/env bash
set -euo pipefail

weights_root="${ITDA_WEIGHTS_ROOT:?Set ITDA_WEIGHTS_ROOT to the Drive weights directory}"
archive="${weights_root}/ppocrv6_medium_rec.tar.gz"
target="${weights_root}/paddle"
test -f "$archive"
mkdir -p "$target"
tar -xzf "$archive" -C "$target"
test -d "$target/PP-OCRv6_medium_rec"
echo "Prepared $target/PP-OCRv6_medium_rec"
