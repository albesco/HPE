#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT_REL="data/intermediate/Side_above_water/_train_canonical"
CONFIG_REL="${DATASET_ROOT_REL}/generated_configs/swimxyz_vitposepp_huge.py"
WORK_DIR_REL="${WORK_DIR_REL:-runs/vitposepp_side_above_water_aniso_20x25_min15}"
PLOTS_OUT_REL="${DATASET_ROOT_REL}/reports/training_plots"
STATUS_FILE_REL="${WORK_DIR_REL}/training_status.txt"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-30}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-3}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
BASE_LOG="${BASE_LOG:-}"
RESUME_FROM="${RESUME_FROM:-${PROJECT_ROOT}/${WORK_DIR_REL}/epoch_4.pth}"

SESSION_NAME="${SESSION_NAME:-vitpose_side_above_water_aniso_resume}"
TS_UTC="$(date -u +%Y%m%d_%H%M%S)"

PLOT_CMD=""
if [[ -n "${BASE_LOG}" ]]; then
  PLOT_CMD=" && LATEST_LOG=\$(ls -1t ${WORK_DIR_REL}/*.log | head -n 1) && \
conda run -n vitpose python script/plot_vitpose_training_log.py \
  --log-file ${BASE_LOG} \
  --log-file \${LATEST_LOG} \
  --output-dir ${PLOTS_OUT_REL} \
  --timestamp ${TS_UTC}"
fi

CMD="cd ${PROJECT_ROOT} && \
conda run -n vitpose python src/vitpose_base/tools/train.py ${CONFIG_REL} \
  --work-dir ${WORK_DIR_REL} \
  --resume-from ${RESUME_FROM} \
  --log-interval 20 \
  --status-file ${STATUS_FILE_REL} \
  --status-interval 20 \
  --final-test \
  --cfg-options total_epochs=${TOTAL_EPOCHS} \
    checkpoint_config.interval=${CHECKPOINT_INTERVAL} \
    checkpoint_config.max_keep_ckpts=${MAX_KEEP_CKPTS} \
    checkpoint_config.create_symlink=True \
    evaluation.interval=${EVAL_INTERVAL}${PLOT_CMD}"

tmux has-session -t "${SESSION_NAME}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
}

tmux new -d -s "${SESSION_NAME}" "${CMD}"
echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Resume from: ${RESUME_FROM}"
