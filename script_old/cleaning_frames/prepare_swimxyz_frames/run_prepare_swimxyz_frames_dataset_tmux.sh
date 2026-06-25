#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vitpose}"
SESSION_NAME="${SESSION_NAME:-prepare_swimxyz_frames_dataset}"
INPUT_ROOT_REL="${INPUT_ROOT_REL:-data/input/subset_xyz/Side_above_water_frames}"
OUTPUT_ROOT_REL="${OUTPUT_ROOT_REL:-data/intermediate/Side_above_water_frames}"
VAL_RATIO="${VAL_RATIO:-0.2}"
TEST_RATIO="${TEST_RATIO:-0.1}"
COPY_MODE="${COPY_MODE:-symlink}"
YOLO_POSE_LINK_MODE="${YOLO_POSE_LINK_MODE:-symlink}"
VITPOSE_WORK_DIR_REL="${VITPOSE_WORK_DIR_REL:-runs/vitposepp_side_above_water_frames}"
LOG_REL="${LOG_REL:-logs/prepare_swimxyz_frames_dataset_$(date -u +%Y%m%d_%H%M%S).log}"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${LOG_REL}")"

TMUX_CMD="cd ${PROJECT_ROOT} && conda run -n ${CONDA_ENV} python script/cleaning_frames/prepare_swimxyz_frames/prepare_swimxyz_frames_dataset.py --input-root ${INPUT_ROOT_REL} --output-root ${OUTPUT_ROOT_REL} --val-ratio ${VAL_RATIO} --test-ratio ${TEST_RATIO} --copy-mode ${COPY_MODE} --vitpose-work-dir ${VITPOSE_WORK_DIR_REL} --yolo-pose-link-mode ${YOLO_POSE_LINK_MODE} --overwrite 2>&1 | tee ${LOG_REL}"

tmux has-session -t "${SESSION_NAME}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
}

tmux new-session -d -s "${SESSION_NAME}" -n rebuild "${TMUX_CMD}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Input root: ${PROJECT_ROOT}/${INPUT_ROOT_REL}"
echo "Output root: ${PROJECT_ROOT}/${OUTPUT_ROOT_REL}"
echo "Log: ${PROJECT_ROOT}/${LOG_REL}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
