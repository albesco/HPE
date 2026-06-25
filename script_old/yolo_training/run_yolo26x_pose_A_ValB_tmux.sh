#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-yolo26x_pose_A_ValB_cpu_20260609}"
OUTDIR="${PROJECT_ROOT}/data/output/experiments/yolo26x-pose_A_20260609"
LOGFILE="${PROJECT_ROOT}/logs/yolo26x_pose_A_ValB_cpu_20260609.log"
TMPJSON="/tmp/yolo26x_pose_A_20260609_valb_json"

mkdir -p "${OUTDIR}" "${PROJECT_ROOT}/logs"
rm -rf "${TMPJSON}"
mkdir -p "${TMPJSON}"

tmux new-session -d -s "${SESSION}" "cd '${PROJECT_ROOT}' && bash -lc 'set -euo pipefail
conda run -n vitpose python script/yolo_training/predict_yolo_pose.py --model runs/yolo26x-pose_A_20260605/weights/best.pt --source data/intermediate/Side_above_water_EntireSwim_B/_train_canonical/val2017 --output-dir ${TMPJSON} --imgsz 768 --conf 0.25 --device cpu
conda run -n vitpose python script/yolo_training/render_yolo_pose_overlays.py --model runs/yolo26x-pose_A_20260605/weights/best.pt --source data/intermediate/Side_above_water_EntireSwim_B/_train_canonical/val2017 --output-dir ${OUTDIR}/overlays_Val_B --imgsz 768 --conf 0.25 --device cpu
python - <<\"PY\"
import json
from pathlib import Path
src = Path(\"${TMPJSON}\")
out = Path(\"${OUTDIR}/keypoints_Val_B.json\")
items = [json.loads(p.read_text()) for p in sorted(src.glob(\"*.json\"))]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(items, indent=2))
PY
' > '${LOGFILE}' 2>&1"

echo "Started tmux session: ${SESSION}"
echo "Attach with: tmux attach -t ${SESSION}"
