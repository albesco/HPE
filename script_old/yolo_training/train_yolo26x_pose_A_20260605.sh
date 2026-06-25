#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RUN_DIR="${PROJECT_ROOT}/runs/yolo26x-pose_A_20260605"
DATA="${PROJECT_ROOT}/data/intermediate/Side_above_water_EntireSwim_A/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml"
MODEL="${MODEL:-${PROJECT_ROOT}/models/pose/yolo26x-pose.pt}"
LOG_DIR="${PROJECT_ROOT}/logs"
TRAIN_LOG="${LOG_DIR}/yolo26x_pose_A_20260605_train.log"
TEST_LOG="${LOG_DIR}/yolo26x_pose_A_20260605_test.log"
MONITOR_STATUS="${RUN_DIR}/monitor_status.json"
TEST_OUT="${PROJECT_ROOT}/data/output/experiments/yolo26x-pose_A_20260605"

mkdir -p "${LOG_DIR}" "${RUN_DIR}" "${TEST_OUT}"

if [[ ! -f "${DATA}" ]]; then
  echo "Missing data YAML: ${DATA}" >&2
  exit 1
fi

setsid bash -lc "cd '${PROJECT_ROOT}' && conda run -n vitpose yolo pose train \
  model='${MODEL}' \
  data='${DATA}' \
  epochs='${EPOCHS:-100}' \
  imgsz=768 \
  lr0=0.00100 \
  batch='${BATCH:-1}' \
  device='${DEVICE:-0}' \
  workers='${WORKERS:-2}' \
  patience=100 \
  save=True \
  save_period=1 \
  val=True \
  split=val \
  plots=True \
  verbose=True \
  seed=0 \
  deterministic=True \
  amp=True \
  resume=False \
  exist_ok=True \
  optimizer=AdamW \
  project='${PROJECT_ROOT}/runs' \
  name='yolo26x-pose_A_20260605' > '${TRAIN_LOG}' 2>&1" &

TRAIN_PID=$!

conda run -n vitpose python "${PROJECT_ROOT}/script/yolo_training/monitor_yolo_pose_patience.py" \
  --run-dir "${RUN_DIR}" \
  --pid "${TRAIN_PID}" \
  --metric "metrics/mAP50-95(P)" \
  --patience 2 \
  --min-delta 0.001 \
  --poll-seconds "${POLL_SECONDS:-60}" \
  --status-json "${MONITOR_STATUS}" 2>&1 | tee -a "${TRAIN_LOG}" &

MONITOR_PID=$!
set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
wait "${MONITOR_PID}"
set -e

if [[ "${TRAIN_STATUS}" -ne 0 && "${TRAIN_STATUS}" -ne 143 ]]; then
  echo "Training failed with exit code ${TRAIN_STATUS}" >&2
  exit "${TRAIN_STATUS}"
fi

BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"
if [[ ! -f "${BEST_WEIGHTS}" ]]; then
  echo "Missing best checkpoint: ${BEST_WEIGHTS}" >&2
  exit 1
fi

conda run -n vitpose python "${PROJECT_ROOT}/script/yolo_training/evaluate_yolo_pose_split.py" \
  --model "${BEST_WEIGHTS}" \
  --data "${DATA}" \
  --split test \
  --imgsz 768 \
  --batch "${BATCH:-1}" \
  --device "${DEVICE:-0}" \
  --workers "${WORKERS:-2}" \
  --out-csv "${TEST_OUT}/best_test_metrics.csv" \
  --out-keypoints-json "${TEST_OUT}/best_test_keypoints.json" \
  --overlays-dir "${TEST_OUT}/overlays_best_test" \
  --overlay-max-images 0 2>&1 | tee "${TEST_LOG}"

echo "run_dir=${RUN_DIR}"
echo "test_output=${TEST_OUT}"
