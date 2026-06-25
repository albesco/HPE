#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vitpose}"

CONFIG_REL="data/intermediate/Side_above_water_EntireSwim_A/_VitPosePP/generated_configs/swimxyz_vitposepp_huge_A_20260606.py"
TRAIN_DATASET_ROOT_REL="data/intermediate/Side_above_water_EntireSwim_A/_train_canonical"
TEST_B_DATASET_ROOT_REL="data/intermediate/Side_above_water_EntireSwim_B/_train_canonical"
WORK_DIR_REL="runs/vitpose_A_20260606"
EXPERIMENT_OUT_REL="data/output/experiments/vitpose_A_20260606"

STATUS_FILE_REL="${WORK_DIR_REL}/training_status.txt"
MONITOR_JSON_REL="${WORK_DIR_REL}/early_stop_status.json"
VAL_METRICS_CSV_REL="${WORK_DIR_REL}/reports/training_plots/val_metrics_by_epoch.csv"
PLOTS_DIR_REL="${WORK_DIR_REL}/reports/training_plots"
TRAIN_PID_FILE_REL="${WORK_DIR_REL}/train_pgid.txt"
TRAIN_STDOUT_REL="${WORK_DIR_REL}/train_stdout.log"
TEST_B_METRICS_REL="${EXPERIMENT_OUT_REL}/test_B_metrics.json"
TEST_B_PRED_REL="${EXPERIMENT_OUT_REL}/test_B_predictions.pkl"
TEST_B_LOG_REL="${EXPERIMENT_OUT_REL}/test_B_eval_stdout.log"
TEST_B_OVERLAYS_REL="${EXPERIMENT_OUT_REL}/overlays_Test_B"

mkdir -p "${PROJECT_ROOT}/${WORK_DIR_REL}" "${PROJECT_ROOT}/${EXPERIMENT_OUT_REL}" "${PROJECT_ROOT}/${PLOTS_DIR_REL}"

TRAIN_CMD="conda run -n '${CONDA_ENV}' python src/vitpose_base/tools/train.py '${CONFIG_REL}' --work-dir '${WORK_DIR_REL}' --log-interval 20 --status-file '${STATUS_FILE_REL}' --status-interval 20"

echo "Starting VitPose++ training"
echo "Config: ${PROJECT_ROOT}/${CONFIG_REL}"
echo "Train dataset root: ${PROJECT_ROOT}/${TRAIN_DATASET_ROOT_REL}"
echo "Test dataset root after training: ${PROJECT_ROOT}/${TEST_B_DATASET_ROOT_REL}"

rm -f "${PROJECT_ROOT}/${TRAIN_PID_FILE_REL}"
(
  cd "${PROJECT_ROOT}"
  setsid bash -lc "cd '${PROJECT_ROOT}'; exec > >(tee '${TRAIN_STDOUT_REL}') 2>&1; echo \$BASHPID > '${PROJECT_ROOT}/${TRAIN_PID_FILE_REL}'; exec ${TRAIN_CMD}"
) &
TRAIN_WRAPPER_PID=$!
sleep 2
if [[ -f "${PROJECT_ROOT}/${TRAIN_PID_FILE_REL}" ]]; then
  TRAIN_PGID="$(tr -d '[:space:]' < "${PROJECT_ROOT}/${TRAIN_PID_FILE_REL}")"
else
  TRAIN_PGID="$(ps -o pgid= "${TRAIN_WRAPPER_PID}" | tr -d ' ')"
fi

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/monitor_vitpose_patience.py" \
  --work-dir "${WORK_DIR_REL}" \
  --pid "${TRAIN_PGID}" \
  --metric AP \
  --patience 2 \
  --keep-last 5 \
  --min-delta 0.005 \
  --poll-interval 60 \
  --status-file "${STATUS_FILE_REL}" \
  --output "${MONITOR_JSON_REL}"

wait "${TRAIN_WRAPPER_PID}" || true

BEST_CKPT="$(find "${PROJECT_ROOT}/${WORK_DIR_REL}" -maxdepth 1 -name 'best_*.pth' | sort -V | tail -n 1)"
if [[ -z "${BEST_CKPT}" && -L "${PROJECT_ROOT}/${WORK_DIR_REL}/latest.pth" ]]; then
  BEST_CKPT="$(readlink -f "${PROJECT_ROOT}/${WORK_DIR_REL}/latest.pth")"
fi
if [[ -z "${BEST_CKPT}" ]]; then
  echo "No checkpoint found in ${WORK_DIR_REL}" >&2
  exit 1
fi

MMPOSE_LOG="$(find "${PROJECT_ROOT}/${WORK_DIR_REL}" -maxdepth 1 -type f -name '*.log' ! -name 'train_stdout.log' | sort | tail -n 1)"
if [[ -z "${MMPOSE_LOG}" ]]; then
  MMPOSE_LOG="${PROJECT_ROOT}/${WORK_DIR_REL}/train_stdout.log"
fi
if [[ ! -f "${MMPOSE_LOG}" ]]; then
  echo "No usable log file found in ${WORK_DIR_REL}" >&2
  exit 1
fi

DEVICE="$(conda run -n "${CONDA_ENV}" python -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')" | tail -n 1 | tr -d '\r')"
echo "Best checkpoint: ${BEST_CKPT}"
echo "Post-train device: ${DEVICE}"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/export_vitpose_val_metrics.py" \
  --work-dir "${WORK_DIR_REL}" \
  --out-csv "${VAL_METRICS_CSV_REL}"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/plot_vitpose_training_log.py" \
  --log-file "${MMPOSE_LOG}" \
  --output-dir "${PLOTS_DIR_REL}" \
  --timestamp "vitpose_A_20260606"

TEST_B_ANN_FILE="${PROJECT_ROOT}/${TEST_B_DATASET_ROOT_REL}/annotations/person_keypoints_test.json"
TEST_B_IMG_PREFIX="${PROJECT_ROOT}/${TEST_B_DATASET_ROOT_REL}/test2017/"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/src/vitpose_base/tools/test.py" \
  "${CONFIG_REL}" \
  "${BEST_CKPT}" \
  --work-dir "${EXPERIMENT_OUT_REL}" \
  --out "${TEST_B_PRED_REL}" \
  --eval mAP \
  --eval-out "${TEST_B_METRICS_REL}" \
  --device "${DEVICE}" \
  --cfg-options \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=1 \
    data.test_dataloader.samples_per_gpu=1 \
    data.test_dataloader.workers_per_gpu=1 \
    data.test.ann_file="${TEST_B_ANN_FILE}" \
    data.test.img_prefix="${TEST_B_IMG_PREFIX}" \
  2>&1 | tee "${TEST_B_LOG_REL}"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/vitpose_generate_test_overlays.py" \
  --dataset-root "${TEST_B_DATASET_ROOT_REL}" \
  --config "${CONFIG_REL}" \
  --checkpoint "${BEST_CKPT}" \
  --output-dir "${TEST_B_OVERLAYS_REL}" \
  --device "${DEVICE}"

echo "Training and Test B evaluation completed."
echo "Work dir: ${PROJECT_ROOT}/${WORK_DIR_REL}"
echo "Experiment outputs: ${PROJECT_ROOT}/${EXPERIMENT_OUT_REL}"
