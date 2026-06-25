#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_REL="data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge_grid_winner_resume.py"
WORK_DIR_REL="${WORK_DIR_REL:-runs/vitposepp_side_above_water_grid_winner_resume}"
STATUS_FILE_REL="${WORK_DIR_REL}/training_status.txt"
MONITOR_OUT_REL="${WORK_DIR_REL}/early_stop_status.json"
CONDA_ENV="${CONDA_ENV:-vitpose}"
SESSION_NAME="${SESSION_NAME:-vitpose_side_above_water_grid_best}"
PATIENCE="${PATIENCE:-5}"

TRAIN_CMD="cd ${PROJECT_ROOT} && conda run -n ${CONDA_ENV} python src/vitpose_base/tools/train.py ${CONFIG_REL} --work-dir ${WORK_DIR_REL} --log-interval 20 --status-file ${STATUS_FILE_REL} --status-interval 20 --final-test"
MONITOR_CMD="cd ${PROJECT_ROOT} && conda run -n ${CONDA_ENV} python script/monitor_vitpose_patience.py --work-dir ${WORK_DIR_REL} --session-name ${SESSION_NAME} --patience ${PATIENCE} --status-file ${STATUS_FILE_REL} --output ${MONITOR_OUT_REL} --kill-session"

tmux has-session -t "${SESSION_NAME}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
}

tmux new-session -d -s "${SESSION_NAME}" -n train "${TRAIN_CMD}"
tmux new-window -t "${SESSION_NAME}":1 -n early-stop "${MONITOR_CMD}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Config: ${PROJECT_ROOT}/${CONFIG_REL}"
echo "Source checkpoint: /home/albertosco/HPE/runs/hparam_search/vitposepp_huge/cfg_02_lr_0.00100_crop_384x128/best_AP_epoch_5.pth"
echo "Patience: ${PATIENCE}"
