#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vitpose}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-auto}"
WAIT_SESSION="${WAIT_SESSION:-vitposepp_test_eval_best24}"
SESSION_NAME="${SESSION_NAME:-vitposepp_test_overlay_after_eval}"
POLL_SECONDS="${POLL_SECONDS:-30}"
CONFIG_REL="${CONFIG_REL:-data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge_grid_winner_resume.py}"
CHECKPOINT_REL="${CHECKPOINT_REL:-runs/vitposepp_side_above_water_grid_winner_resume/best_AP_epoch_24.pth}"
OUTPUT_DIR_REL="${OUTPUT_DIR_REL:-data/intermediate/Side_above_water/_train_canonical/reports/test_overlays/vitposepp_grid_winner_best_AP_epoch_24}"
LOG_REL="${LOG_REL:-${OUTPUT_DIR_REL}/overlay_generation_stdout.log}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR_REL}"

WAIT_AND_RUN_CMD="cd ${PROJECT_ROOT} && while tmux has-session -t ${WAIT_SESSION} 2>/dev/null; do sleep ${POLL_SECONDS}; done; conda run -n ${CONDA_ENV} python script/vitpose_generate_test_overlays.py --config ${CONFIG_REL} --checkpoint ${CHECKPOINT_REL} --output-dir ${OUTPUT_DIR_REL} --device ${DEVICE} 2>&1 | tee ${LOG_REL}"

tmux has-session -t "${SESSION_NAME}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
}

tmux new-session -d -s "${SESSION_NAME}" -n wait-overlay "${WAIT_AND_RUN_CMD}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Waiting for session to finish: ${WAIT_SESSION}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Overlay output dir: ${PROJECT_ROOT}/${OUTPUT_DIR_REL}"
echo "Overlay log: ${PROJECT_ROOT}/${LOG_REL}"
