#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vitpose}"
SESSION_NAME="${SESSION_NAME:-prepare_entire_swim_dataset}"
COPY_MODE="${COPY_MODE:-symlink}"
YOLO_POSE_LINK_MODE="${YOLO_POSE_LINK_MODE:-symlink}"
VITPOSE_WORK_DIR_REL="${VITPOSE_WORK_DIR_REL:-runs/vitposepp_side_above_water_entireswim}"
OUTPUT_DATASET_ROOT_REL="${OUTPUT_DATASET_ROOT_REL:-data/intermediate/Side_above_water_EntireSwim}"
SOURCE_DATASET_ROOTS_REL="${SOURCE_DATASET_ROOTS_REL:-data/intermediate/Side_above_water data/intermediate/Side_above_water_VideoTest2}"
LOG_REL="${LOG_REL:-logs/prepare_entire_swim_dataset_$(date -u +%Y%m%d_%H%M%S).log}"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${LOG_REL}")"

TMUX_CMD="cd ${PROJECT_ROOT} && conda run -n ${CONDA_ENV} python script/prepare_entire_swim_dataset.py --source-dataset-roots ${SOURCE_DATASET_ROOTS_REL} --output-dataset-root ${OUTPUT_DATASET_ROOT_REL} --copy-mode ${COPY_MODE} --vitpose-work-dir ${VITPOSE_WORK_DIR_REL} --yolo-pose-link-mode ${YOLO_POSE_LINK_MODE} --overwrite 2>&1 | tee ${LOG_REL}"

tmux has-session -t "${SESSION_NAME}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
}

tmux new-session -d -s "${SESSION_NAME}" -n rebuild "${TMUX_CMD}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Output dataset root: ${PROJECT_ROOT}/${OUTPUT_DATASET_ROOT_REL}"
echo "Log: ${PROJECT_ROOT}/${LOG_REL}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
