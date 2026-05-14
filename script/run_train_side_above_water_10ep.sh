#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT_REL="data/intermediate/Side_above_water/_train_vitposepp_swap_ears"
CONFIG_REL="${DATASET_ROOT_REL}/generated_configs/swimxyz_vitposepp_huge_single_head_swap_ears.py"
WORK_DIR_REL="runs/vitposepp_single_head_subset_xyz_swap_ears"
PLOTS_OUT_REL="${DATASET_ROOT_REL}/reports/training_plots"
STATUS_FILE_REL="${WORK_DIR_REL}/training_status.txt"

TS_UTC="$(date -u +%Y%m%d_%H%M%S)"

conda run -n vitpose python "${PROJECT_ROOT}/src/vitpose_base/tools/train.py" \
  "${PROJECT_ROOT}/${CONFIG_REL}" \
  --log-interval 20 \
  --status-file "${PROJECT_ROOT}/${STATUS_FILE_REL}" \
  --status-interval 20 \
  --cfg-options total_epochs=10 checkpoint_config.interval=5 evaluation.interval=5

LATEST_LOG="$(ls -1t "${PROJECT_ROOT}/${WORK_DIR_REL}"/*.log | head -n 1)"

conda run -n vitpose python "${PROJECT_ROOT}/script/plot_vitpose_training_log.py" \
  --log-file "${LATEST_LOG}" \
  --output-dir "${PROJECT_ROOT}/${PLOTS_OUT_REL}" \
  --timestamp "${TS_UTC}"

echo "Done. Plots: ${PROJECT_ROOT}/${PLOTS_OUT_REL}"
