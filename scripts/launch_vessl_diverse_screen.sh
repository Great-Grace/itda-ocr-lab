#!/usr/bin/env bash
# Run the Stage-A recognizer architecture screen with two independent workers
# per A100.  It deliberately does not perform final inference on GPU.
set -euo pipefail

TASK_PROJECT_ROOT=${1:-/workspace/itda-ocr}
TASK_DATA_ROOT=${2:-/data/itda}
TASK_EPOCHS=${3:-12}
TASK_BATCH_SIZE=${4:-128}
TASK_RUN_ID=${5:-"diverse_screen_$(date -u +%Y%m%dT%H%M%SZ)"}
TASK_PYTHON=${TASK_PYTHON:-python}
TASK_SPECS="${TASK_SPECS:-$TASK_PROJECT_ROOT/configs/recognizer_suites/diverse_screen_v1.json}"
TASK_OUTPUT="$TASK_DATA_ROOT/artifacts/$TASK_RUN_ID"
TASK_TRAIN_CROPS="${TASK_TRAIN_CROPS:-$TASK_DATA_ROOT/artifacts/real_crops_train}"
TASK_VAL_CROPS="${TASK_VAL_CROPS:-$TASK_DATA_ROOT/artifacts/real_crops_val}"
export PYTHONPATH="$TASK_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

for TASK_REQUIRED in "$TASK_SPECS" "$TASK_TRAIN_CROPS/labels.csv" "$TASK_VAL_CROPS/labels.csv"; do
  test -f "$TASK_REQUIRED" || { echo "Missing required input: $TASK_REQUIRED" >&2; exit 1; }
done
mkdir -p "$TASK_OUTPUT/spec_shards" "$TASK_OUTPUT/logs"

"$TASK_PYTHON" - "$TASK_SPECS" "$TASK_OUTPUT/spec_shards" <<'PY'
import json
import sys
from pathlib import Path

specs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
for index in range(4):
    shard = specs[index::4]
    (out / f"worker_{index}.json").write_text(json.dumps(shard, ensure_ascii=False, indent=2), encoding="utf-8")
PY

run_worker() {
  local task_gpu=$1
  local task_worker=$2
  CUDA_VISIBLE_DEVICES="$task_gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$TASK_PYTHON" "$TASK_PROJECT_ROOT/scripts/train_real_crop_suite.py" \
      --train-crops "$TASK_TRAIN_CROPS" \
      --val-crops "$TASK_VAL_CROPS" \
      --architecture-specs "$TASK_OUTPUT/spec_shards/worker_${task_worker}.json" \
      --output "$TASK_OUTPUT/worker_${task_worker}" \
      --device cuda --epochs "$TASK_EPOCHS" --batch-size "$TASK_BATCH_SIZE" \
      > "$TASK_OUTPUT/logs/worker_${task_worker}.log" 2>&1
}

# Independent processes avoid DDP coordination overhead for these small models.
run_worker 0 0 & TASK_PID_0=$!
run_worker 0 1 & TASK_PID_1=$!
run_worker 1 2 & TASK_PID_2=$!
run_worker 1 3 & TASK_PID_3=$!

TASK_STATUS=0
for TASK_PID in "$TASK_PID_0" "$TASK_PID_1" "$TASK_PID_2" "$TASK_PID_3"; do
  wait "$TASK_PID" || TASK_STATUS=1
done

"$TASK_PYTHON" - "$TASK_OUTPUT" "$TASK_EPOCHS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
epochs = int(sys.argv[2])
rows = []
for path in sorted(root.glob("worker_*/suite_summary.json")):
    rows.extend(json.loads(path.read_text(encoding="utf-8")).get("architectures", []))
payload = {"stage": "A", "screening_epochs": epochs, "architectures": rows}
(root / "suite_summary_merged.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"completed_architectures": len(rows), "output": str(root)}, ensure_ascii=False))
PY

exit "$TASK_STATUS"
