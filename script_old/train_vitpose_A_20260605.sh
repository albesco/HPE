#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vitpose}"

CONFIG_REL="data/intermediate/Side_above_water_EntireSwim_A/_VitPosePP/generated_configs/swimxyz_vitposepp_huge_A_20260605.py"
DATASET_ROOT_REL="data/intermediate/Side_above_water_EntireSwim_A/_train_canonical"
WORK_DIR_REL="runs/vitpose_A_20260605"
EXPERIMENT_OUT_REL="data/output/experiments/vitpose_A_20260605"

STATUS_FILE_REL="${WORK_DIR_REL}/training_status.txt"
MONITOR_JSON_REL="${WORK_DIR_REL}/early_stop_status.json"
VAL_METRICS_CSV_REL="${WORK_DIR_REL}/val_metrics_by_epoch.csv"
TRAIN_LOG_REL="${WORK_DIR_REL}/train_stdout.log"
PLOTS_DIR_REL="${WORK_DIR_REL}/reports/training_plots"
TRAIN_PID_FILE_REL="${WORK_DIR_REL}/train_pgid.txt"
GT_TEST_OUT_REL="${EXPERIMENT_OUT_REL}/gt_bbox_best_test"
GT_TEST_LOG_REL="${GT_TEST_OUT_REL}/test_eval_stdout.log"
GT_TEST_METRICS_REL="${GT_TEST_OUT_REL}/test_metrics.json"
GT_TEST_PRED_REL="${GT_TEST_OUT_REL}/test_predictions.pkl"
GT_OVERLAYS_REL="${GT_TEST_OUT_REL}/overlays"

mkdir -p "${PROJECT_ROOT}/${WORK_DIR_REL}" "${PROJECT_ROOT}/${EXPERIMENT_OUT_REL}" "${PROJECT_ROOT}/${GT_TEST_OUT_REL}" "${PROJECT_ROOT}/${PLOTS_DIR_REL}"

TRAIN_CMD="conda run -n '${CONDA_ENV}' python src/vitpose_base/tools/train.py '${CONFIG_REL}' --work-dir '${WORK_DIR_REL}' --log-interval 20 --status-file '${STATUS_FILE_REL}' --status-interval 20"

echo "Starting VitPose++ training"
echo "Config: ${PROJECT_ROOT}/${CONFIG_REL}"
echo "Work dir: ${PROJECT_ROOT}/${WORK_DIR_REL}"
echo "BBox source during training: GT dataset bboxes"

rm -f "${PROJECT_ROOT}/${TRAIN_PID_FILE_REL}"
(
  cd "${PROJECT_ROOT}"
  setsid bash -lc "cd '${PROJECT_ROOT}'; exec > >(tee '${TRAIN_LOG_REL}') 2>&1; echo \$BASHPID > '${PROJECT_ROOT}/${TRAIN_PID_FILE_REL}'; exec ${TRAIN_CMD}"
) &
TRAIN_WRAPPER_PID=$!
sleep 2
if [[ -f "${PROJECT_ROOT}/${TRAIN_PID_FILE_REL}" ]]; then
  TRAIN_PGID="$(tr -d '[:space:]' < "${PROJECT_ROOT}/${TRAIN_PID_FILE_REL}")"
else
  TRAIN_PGID="$(ps -o pgid= "${TRAIN_WRAPPER_PID}" | tr -d ' ')"
fi

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/monitor_vitpose_patience.py"   --work-dir "${WORK_DIR_REL}"   --pid "${TRAIN_PGID}"   --metric AP   --patience 2   --min-delta 0.001   --poll-interval 60   --status-file "${STATUS_FILE_REL}"   --output "${MONITOR_JSON_REL}"

wait "${TRAIN_WRAPPER_PID}" || true

BEST_CKPT="$(find "${PROJECT_ROOT}/${WORK_DIR_REL}" -maxdepth 1 -name 'best_*.pth' | sort -V | tail -n 1)"
if [[ -z "${BEST_CKPT}" && -L "${PROJECT_ROOT}/${WORK_DIR_REL}/latest.pth" ]]; then
  BEST_CKPT="$(readlink -f "${PROJECT_ROOT}/${WORK_DIR_REL}/latest.pth")"
fi
if [[ -z "${BEST_CKPT}" ]]; then
  echo "No checkpoint found in ${WORK_DIR_REL}" >&2
  exit 1
fi

DEVICE="$(conda run -n "${CONDA_ENV}" python -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')" | tail -n 1 | tr -d '
')"
echo "Best checkpoint: ${BEST_CKPT}"
echo "Post-train device: ${DEVICE}"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/export_vitpose_val_metrics.py"   --work-dir "${WORK_DIR_REL}"   --out-csv "${VAL_METRICS_CSV_REL}"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/plot_vitpose_training_log.py"   --log-file "${TRAIN_LOG_REL}"   --output-dir "${PLOTS_DIR_REL}"   --timestamp "vitpose_A_20260605"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/src/vitpose_base/tools/test.py"   "${CONFIG_REL}"   "${BEST_CKPT}"   --work-dir "${GT_TEST_OUT_REL}"   --out "${GT_TEST_PRED_REL}"   --eval mAP   --eval-out "${GT_TEST_METRICS_REL}"   --device "${DEVICE}"   --cfg-options data.samples_per_gpu=1 data.workers_per_gpu=1 data.test_dataloader.samples_per_gpu=1 data.test_dataloader.workers_per_gpu=1   2>&1 | tee "${GT_TEST_LOG_REL}"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/vitpose_generate_test_overlays.py"   --dataset-root "${DATASET_ROOT_REL}"   --config "${CONFIG_REL}"   --checkpoint "${BEST_CKPT}"   --output-dir "${GT_OVERLAYS_REL}"   --device "${DEVICE}"

echo "Training and GT-bbox post-train evaluation completed."
echo "Work dir: ${PROJECT_ROOT}/${WORK_DIR_REL}"
echo "Experiment outputs: ${PROJECT_ROOT}/${EXPERIMENT_OUT_REL}"
