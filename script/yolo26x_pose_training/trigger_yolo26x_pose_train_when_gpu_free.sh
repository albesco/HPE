#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION_NAME="${SESSION_NAME:-trigger_yolo26x_pose_SUW_frames_20260705_retry1}"
CONDA_ENV="${CONDA_ENV:-vitpose}"
DEVICE="${DEVICE:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-28}"
POLL_SECONDS="${POLL_SECONDS:-120}"
DATASET_DIR="${DATASET_DIR:-data/intermediate/SUW_frames}"
RUN_NAME="${RUN_NAME:-SUW_frames_20260705_retry1}"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SESSION_NAME}.log}"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" -x 200 -y 50 \
  "cd '${PROJECT_ROOT}' && \
   while true; do \
     FREE_GB=\$(conda run -n '${CONDA_ENV}' python script/yolo26x_pose_training/cuda_free_gb.py --device '${DEVICE}'); \
     TS=\$(date -u +'%Y-%m-%dT%H:%M:%SZ'); \
     echo \"\${TS} free_gpu_gb=\${FREE_GB} threshold_gb=${MIN_FREE_GB}\" | tee -a '${LOG_FILE}'; \
     awk -v free=\"\${FREE_GB}\" -v min='${MIN_FREE_GB}' 'BEGIN { exit !(free >= min) }' && break; \
     sleep '${POLL_SECONDS}'; \
   done; \
   echo \"\$(date -u +'%Y-%m-%dT%H:%M:%SZ') launching YOLO26x-Pose train\" | tee -a '${LOG_FILE}'; \
   bash script/yolo26x_pose_training/train_yolo26x_pose_frame.sh \
     --dataset-dir '${DATASET_DIR}' \
     --run-name '${RUN_NAME}' 2>&1 | tee -a '${LOG_FILE}'"

echo "Started tmux trigger: ${SESSION_NAME}"
echo "Log: ${LOG_FILE}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
