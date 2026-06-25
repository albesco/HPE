#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR="${PROJECT_ROOT}/data/output/experiments/yolo26x-pose_A_20260609"
TMPJSON="/tmp/yolo26x_pose_A_20260609_valb_json"
MODEL="${PROJECT_ROOT}/runs/yolo26x-pose_A_20260605/weights/best.pt"
SOURCE="${SOURCE:-${PROJECT_ROOT}/data/intermediate/Side_above_water_EntireSwim_B/_train_canonical/val2017}"
DEVICE="${DEVICE:-0}"

mkdir -p "${OUTDIR}" "${PROJECT_ROOT}/logs"
rm -rf "${TMPJSON}"
mkdir -p "${TMPJSON}"

conda run -n vitpose python "${PROJECT_ROOT}/script/yolo_training/predict_yolo_pose.py" \
  --model "${MODEL}" \
  --source "${SOURCE}" \
  --output-dir "${TMPJSON}" \
  --imgsz 768 \
  --conf 0.25 \
  --device "${DEVICE}"

conda run -n vitpose python "${PROJECT_ROOT}/script/yolo_training/render_yolo_pose_overlays.py" \
  --model "${MODEL}" \
  --source "${SOURCE}" \
  --output-dir "${OUTDIR}/overlays_Val_B" \
  --imgsz 768 \
  --conf 0.25 \
  --device "${DEVICE}"

TMPJSON="${TMPJSON}" OUTFILE="${OUTDIR}/keypoints_Val_B.json" python - <<'PY'
import json
import os
from pathlib import Path
src = Path(os.environ['TMPJSON'])
out = Path(os.environ['OUTFILE'])
items = [json.loads(p.read_text()) for p in sorted(src.glob('*.json'))]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(items, indent=2))
PY

echo "keypoints=${OUTDIR}/keypoints_Val_B.json"
echo "overlays=${OUTDIR}/overlays_Val_B"
