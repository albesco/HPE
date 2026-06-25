#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vitpose}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-auto}"
SESSION_NAME="${SESSION_NAME:-vitposepp_test_eval_best24}"
CONFIG_REL="${CONFIG_REL:-data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge_grid_winner_resume.py}"
CHECKPOINT_REL="${CHECKPOINT_REL:-runs/vitposepp_side_above_water_grid_winner_resume/best_AP_epoch_24.pth}"
OUT_DIR_REL="${OUT_DIR_REL:-runs/vitposepp_side_above_water_grid_winner_resume/test_eval_best_AP_epoch_24}"
PREDICTIONS_REL="${PREDICTIONS_REL:-${OUT_DIR_REL}/test_predictions.pkl}"
LOG_REL="${LOG_REL:-${OUT_DIR_REL}/test_eval_stdout.log}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_DIR_REL}"

TEST_CMD="cd ${PROJECT_ROOT} && conda run -n ${CONDA_ENV} python src/vitpose_base/tools/test.py ${CONFIG_REL} ${CHECKPOINT_REL} --work-dir ${OUT_DIR_REL} --out ${PREDICTIONS_REL} --eval mAP --device ${DEVICE} --gpu-id ${GPU_ID} --cfg-options data.samples_per_gpu=1 data.workers_per_gpu=1 data.test_dataloader.samples_per_gpu=1 data.test_dataloader.workers_per_gpu=1 2>&1 | tee ${LOG_REL}"

tmux has-session -t "${SESSION_NAME}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
}

tmux new-session -d -s "${SESSION_NAME}" -n test-eval "${TEST_CMD}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Config: ${PROJECT_ROOT}/${CONFIG_REL}"
echo "Checkpoint: ${PROJECT_ROOT}/${CHECKPOINT_REL}"
echo "Out dir: ${PROJECT_ROOT}/${OUT_DIR_REL}"
echo "Predictions: ${PROJECT_ROOT}/${PREDICTIONS_REL}"
echo "Device: ${DEVICE}"
echo "GPU: ${GPU_ID}"
