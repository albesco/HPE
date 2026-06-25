#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vitpose}"
SESSION_NAME="${SESSION_NAME:-prepare_entire_swim_ab_split}"
SOURCE_DATASET_ROOT_REL="${SOURCE_DATASET_ROOT_REL:-data/intermediate/Side_above_water_EntireSwim}"
OUTPUT_DATASET_ROOT_A_REL="${OUTPUT_DATASET_ROOT_A_REL:-data/intermediate/Side_above_water_EntireSwim_A}"
OUTPUT_DATASET_ROOT_B_REL="${OUTPUT_DATASET_ROOT_B_REL:-data/intermediate/Side_above_water_EntireSwim_B}"
PROB_A="${PROB_A:-0.3}"
SEED="${SEED:-20260604}"
COPY_MODE="${COPY_MODE:-symlink}"
YOLO_POSE_LINK_MODE="${YOLO_POSE_LINK_MODE:-symlink}"
VITPOSE_WORK_DIR_A_REL="${VITPOSE_WORK_DIR_A_REL:-runs/vitposepp_side_above_water_entireswim_a}"
VITPOSE_WORK_DIR_B_REL="${VITPOSE_WORK_DIR_B_REL:-runs/vitposepp_side_above_water_entireswim_b}"
LOG_REL="${LOG_REL:-logs/prepare_entire_swim_ab_split_$(date -u +%Y%m%d_%H%M%S).log}"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${LOG_REL}")"

TMUX_CMD="cd ${PROJECT_ROOT} && conda run -n ${CONDA_ENV} python script/prepare_entire_swim_ab_split.py --source-dataset-root ${SOURCE_DATASET_ROOT_REL} --output-dataset-root-a ${OUTPUT_DATASET_ROOT_A_REL} --output-dataset-root-b ${OUTPUT_DATASET_ROOT_B_REL} --prob-a ${PROB_A} --seed ${SEED} --copy-mode ${COPY_MODE} --vitpose-work-dir-a ${VITPOSE_WORK_DIR_A_REL} --vitpose-work-dir-b ${VITPOSE_WORK_DIR_B_REL} --yolo-pose-link-mode ${YOLO_POSE_LINK_MODE} --overwrite 2>&1 | tee ${LOG_REL}"

tmux has-session -t "${SESSION_NAME}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
}

tmux new-session -d -s "${SESSION_NAME}" -n rebuild "${TMUX_CMD}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Source dataset root: ${PROJECT_ROOT}/${SOURCE_DATASET_ROOT_REL}"
echo "Output dataset root A: ${PROJECT_ROOT}/${OUTPUT_DATASET_ROOT_A_REL}"
echo "Output dataset root B: ${PROJECT_ROOT}/${OUTPUT_DATASET_ROOT_B_REL}"
echo "Log: ${PROJECT_ROOT}/${LOG_REL}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
