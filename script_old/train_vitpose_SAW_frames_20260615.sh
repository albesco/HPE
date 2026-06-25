#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vitpose}"

CONFIG_REL="${CONFIG_REL:-data/intermediate/SAW_frames/_VitPosePP/generated_configs/swimxyz_vitposepp_huge_SAW_frames_20260615.py}"
TRAIN_DATASET_ROOT_REL="${TRAIN_DATASET_ROOT_REL:-data/intermediate/SAW_frames/_train_canonical}"
TEST_DATASET_ROOT_REL="${TEST_DATASET_ROOT_REL:-${TRAIN_DATASET_ROOT_REL}}"
RUN_ROOT_REL="runs/vitpose_SAW_frames_20260615"
WORK_DIR_REL="${RUN_ROOT_REL}/checkpoint"
REPORTS_DIR_REL="${RUN_ROOT_REL}/reports"
OUTPUT_DIR_REL="data/output/experiments/vitpose_SAW_frames_20260615"

STATUS_FILE_REL="${RUN_ROOT_REL}/training_status.txt"
MONITOR_JSON_REL="${RUN_ROOT_REL}/early_stop_status.json"
VAL_METRICS_CSV_REL="${REPORTS_DIR_REL}/val_metrics_by_epoch.csv"
TRAIN_PID_FILE_REL="${RUN_ROOT_REL}/train_pgid.txt"
TRAIN_STDOUT_REL="${RUN_ROOT_REL}/train_stdout.log"
TEST_METRICS_REL="${OUTPUT_DIR_REL}/metrics_Test.json"
KP_JSON_REL="${OUTPUT_DIR_REL}/kp_Test.json"
TEST_OVERLAYS_REL="${OUTPUT_DIR_REL}/overlays_Test"

show_help() {
  cat <<'HELP'
Usage:
  bash script/train_vitpose_SAW_frames_20260615.sh [OPTIONS]

Options:
  --config <path>
  --train-dataset-root <dir>
  --test-dataset-root <dir>   Default: same as --train-dataset-root
  --help
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_REL="$2"; shift 2 ;;
    --train-dataset-root) TRAIN_DATASET_ROOT_REL="$2"; shift 2 ;;
    --test-dataset-root) TEST_DATASET_ROOT_REL="$2"; shift 2 ;;
    --help|-h) show_help; exit 0 ;;
    *) echo "Unknown option: $1" >&2; show_help; exit 1 ;;
  esac
done

resolve_project_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${PROJECT_ROOT}" "$1" ;;
  esac
}

CONFIG_PATH="$(resolve_project_path "${CONFIG_REL}")"
TRAIN_DATASET_ROOT="$(resolve_project_path "${TRAIN_DATASET_ROOT_REL}")"
TEST_DATASET_ROOT="$(resolve_project_path "${TEST_DATASET_ROOT_REL}")"

TRAIN_ANN_FILE="${TRAIN_DATASET_ROOT}/annotations/person_keypoints_train.json"
TRAIN_IMG_PREFIX="${TRAIN_DATASET_ROOT}/train2017/"
VAL_ANN_FILE="${TRAIN_DATASET_ROOT}/annotations/person_keypoints_val.json"
VAL_IMG_PREFIX="${TRAIN_DATASET_ROOT}/val2017/"
TEST_ANN_FILE="${TEST_DATASET_ROOT}/annotations/person_keypoints_test.json"
TEST_IMG_PREFIX="${TEST_DATASET_ROOT}/test2017/"

mkdir -p \
  "${PROJECT_ROOT}/${WORK_DIR_REL}" \
  "${PROJECT_ROOT}/${REPORTS_DIR_REL}" \
  "${PROJECT_ROOT}/${OUTPUT_DIR_REL}"

TRAIN_CMD="conda run -n '${CONDA_ENV}' python src/vitpose_base/tools/train.py '${CONFIG_PATH}' --work-dir '${WORK_DIR_REL}' --log-interval 20 --status-file '${STATUS_FILE_REL}' --status-interval 20 --cfg-options data.train.ann_file='${TRAIN_ANN_FILE}' data.train.img_prefix='${TRAIN_IMG_PREFIX}' data.val.ann_file='${VAL_ANN_FILE}' data.val.img_prefix='${VAL_IMG_PREFIX}'"

echo "Starting VitPose++ training"
echo "Script: ${PROJECT_ROOT}/script/train_vitpose_SAW_frames_20260615.sh"
echo "Config: ${CONFIG_PATH}"
echo "Train dataset root: ${TRAIN_DATASET_ROOT}"
echo "Test dataset root after training: ${TEST_DATASET_ROOT}"

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
  --patience 3 \
  --keep-last 10 \
  --min-delta 0.007 \
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
  MMPOSE_LOG="${PROJECT_ROOT}/${TRAIN_STDOUT_REL}"
fi
if [[ ! -f "${MMPOSE_LOG}" ]]; then
  echo "No usable log file found in ${WORK_DIR_REL}" >&2
  exit 1
fi

DEVICE="$(conda run -n "${CONDA_ENV}" python -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')" | tail -n 1 | tr -d '\r')"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/export_vitpose_val_metrics.py" \
  --work-dir "${WORK_DIR_REL}" \
  --out-csv "${VAL_METRICS_CSV_REL}"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/plot_vitpose_training_log.py" \
  --log-file "${MMPOSE_LOG}" \
  --output-dir "${REPORTS_DIR_REL}" \
  --timestamp "vitpose_SAW_frames_20260615"

rm -f "${PROJECT_ROOT}/${OUTPUT_DIR_REL}/result_keypoints.json" "${PROJECT_ROOT}/${KP_JSON_REL}" "${PROJECT_ROOT}/${TEST_METRICS_REL}"
rm -rf "${PROJECT_ROOT}/${TEST_OVERLAYS_REL}"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/src/vitpose_base/tools/test.py" \
  "${CONFIG_PATH}" \
  "${BEST_CKPT}" \
  --work-dir "${PROJECT_ROOT}/${OUTPUT_DIR_REL}" \
  --eval mAP \
  --eval-out "${PROJECT_ROOT}/${TEST_METRICS_REL}" \
  --device "${DEVICE}" \
  --cfg-options \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=1 \
    data.test_dataloader.samples_per_gpu=1 \
    data.test_dataloader.workers_per_gpu=1 \
    data.test.ann_file="${TEST_ANN_FILE}" \
    data.test.img_prefix="${TEST_IMG_PREFIX}"

cp "${PROJECT_ROOT}/${OUTPUT_DIR_REL}/result_keypoints.json" "${PROJECT_ROOT}/${KP_JSON_REL}"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/vitpose_generate_test_overlays_from_json.py" \
  --dataset-root "${TEST_DATASET_ROOT}" \
  --split test \
  --predictions-json "${PROJECT_ROOT}/${KP_JSON_REL}" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${BEST_CKPT}" \
  --output-dir "${PROJECT_ROOT}/${TEST_OVERLAYS_REL}" \
  --device "${DEVICE}"

echo "Training and Test outputs completed."
