#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL="${MODEL:-${PROJECT_ROOT}/models/detection/yolo26x.pt}"
DATA="${DATA:-${PROJECT_ROOT}/data/intermediate/Side_above_water/_yolo_detection/swimxyz_side_above_water_yolo.yaml}"
PROJECT="${PROJECT:-${PROJECT_ROOT}/runs/yolo_side_above_water}"
NAME="${NAME:-yolo26x_swimmer_gt_bbox}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-1280}"
BATCH="${BATCH:-2}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-2}"
PATIENCE="${PATIENCE:-30}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/yolo_side_above_water_${TIMESTAMP}.log}"

mkdir -p "${LOG_DIR}"

echo "Logging YOLO training to: ${LOG_PATH}"
conda run -n vitpose yolo detect train \
  model="${MODEL}" \
  data="${DATA}" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  workers="${WORKERS}" \
  patience="${PATIENCE}" \
  project="${PROJECT}" \
  name="${NAME}" 2>&1 | tee "${LOG_PATH}"
