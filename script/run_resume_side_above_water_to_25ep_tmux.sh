#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT_REL="data/intermediate/Side_above_water/_train_vitposepp_swap_ears"
CONFIG_REL="${DATASET_ROOT_REL}/generated_configs/swimxyz_vitposepp_huge_single_head_swap_ears.py"
WORK_DIR_REL="runs/vitposepp_single_head_subset_xyz_swap_ears"
PLOTS_OUT_REL="${DATASET_ROOT_REL}/reports/training_plots"
STATUS_FILE_REL="${WORK_DIR_REL}/training_status.txt"

BASE_LOG="${PROJECT_ROOT}/${WORK_DIR_REL}/20260512_142708.log"
RESUME_FROM="${PROJECT_ROOT}/${WORK_DIR_REL}/epoch_10.pth"

SESSION_NAME="vitpose_side_above_water_ep25"
TS_UTC="$(date -u +%Y%m%d_%H%M%S)"

CMD="cd ${PROJECT_ROOT} && \
conda run -n vitpose python src/vitpose_base/tools/train.py ${CONFIG_REL} \
  --resume-from ${RESUME_FROM} \
  --log-interval 20 \
  --status-file ${STATUS_FILE_REL} \
  --status-interval 20 \
  --cfg-options total_epochs=25 checkpoint_config.interval=5 evaluation.interval=5 && \
LATEST_LOG=\$(ls -1t ${WORK_DIR_REL}/*.log | head -n 1) && \
conda run -n vitpose python script/plot_vitpose_training_log.py \
  --log-file ${BASE_LOG} \
  --log-file \${LATEST_LOG} \
  --output-dir ${PLOTS_OUT_REL} \
  --timestamp ${TS_UTC}"

tmux has-session -t "${SESSION_NAME}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
}

tmux new -d -s "${SESSION_NAME}" "${CMD}"
echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
