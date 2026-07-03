#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt}"
MODEL="${MODEL:-${BASE_MODEL}}"
DATA="${DATA:-${PROJECT_ROOT}/data/intermediate/Side_above_water/_Yolo26x_detection/swimxyz_side_above_water_yolo26x_detection.yaml}"
PROJECT="${PROJECT:-${PROJECT_ROOT}/runs/yolo26x_bbox_side_above_water}"
TAG="${TAG:-cfg03_lr0_0.00067_imgsz_768_incremental}"
NAME="${NAME:-yolo26x-detection_${TAG}}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-768}"
BATCH="${BATCH:-2}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-2}"
PATIENCE="${PATIENCE:-2}"
LR0="${LR0:-0.00067}"
SAVE_PERIOD="${SAVE_PERIOD:-1}"
KEEP_EPOCH_CKPTS="${KEEP_EPOCH_CKPTS:-3}"
SEED="${SEED:-0}"
DETERMINISTIC="${DETERMINISTIC:-True}"
AMP="${AMP:-True}"
RESUME="${RESUME:-auto}"
EXIST_OK="${EXIST_OK:-True}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_DIR="${PROJECT}/${NAME}"
WEIGHTS_DIR="${RUN_DIR}/weights"
STATUS_FILE="${STATUS_FILE:-${RUN_DIR}/training_status.txt}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/yolo26x_detection_side_above_water_${TIMESTAMP}.log}"

RUN_TEST="${RUN_TEST:-False}"
TEST_SPLIT="${TEST_SPLIT:-test}"
TEST_WEIGHTS="${TEST_WEIGHTS:-best}"
TEST_LOG_PATH="${TEST_LOG_PATH:-${LOG_DIR}/yolo26x_detection_test_eval_${TIMESTAMP}.log}"

write_status() {
  local phase="$1"
  local exit_code="${2:-}"
  mkdir -p "$(dirname "${STATUS_FILE}")"
  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "phase=${phase}"
    [[ -n "${exit_code}" ]] && echo "exit_code=${exit_code}"
    echo "run_dir=${RUN_DIR}"
    echo "weights_dir=${WEIGHTS_DIR}"
    echo "log_path=${LOG_PATH}"
    echo "base_model=${BASE_MODEL}"
    echo "model=${MODEL}"
    echo "data=${DATA}"
    echo "epochs=${EPOCHS}"
    echo "imgsz=${IMGSZ}"
    echo "batch=${BATCH}"
    echo "patience=${PATIENCE}"
    echo "lr0=${LR0}"
    echo "save_period=${SAVE_PERIOD}"
    echo "keep_epoch_ckpts=${KEEP_EPOCH_CKPTS}"
    echo "resume=${RESUME}"
    echo "run_test=${RUN_TEST}"
    echo "test_split=${TEST_SPLIT}"
    echo "test_weights=${TEST_WEIGHTS}"
    echo "test_log_path=${TEST_LOG_PATH}"
  } > "${STATUS_FILE}"
}

if [[ ! -f "${DATA}" ]]; then
  echo "Missing YOLO26x detection data YAML: ${DATA}" >&2
  echo "Prepare it first with: conda run -n vitpose python script/dataset_preparation-cleaning/prepare_yolo_detection_dataset.py --overwrite" >&2
  exit 1
fi

if [[ "${RESUME}" == "auto" ]]; then
  if [[ -f "${WEIGHTS_DIR}/last.pt" ]]; then
    MODEL="${WEIGHTS_DIR}/last.pt"
    RESUME="True"
  else
    RESUME="False"
  fi
fi

if [[ "${MODEL}" = /* && ! -f "${MODEL}" ]]; then
  echo "Missing YOLO26x detection model weights: ${MODEL}" >&2
  echo "Expected the selected cfg_03 checkpoint at: ${BASE_MODEL}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${RUN_DIR}"
write_status "starting"

echo "Logging YOLO26x detection training to: ${LOG_PATH}"
set +e
conda run -n vitpose yolo detect train \
  model="${MODEL}" \
  data="${DATA}" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  workers="${WORKERS}" \
  patience="${PATIENCE}" \
  lr0="${LR0}" \
  save=True \
  save_period="${SAVE_PERIOD}" \
  val=True \
  plots=True \
  verbose=True \
  seed="${SEED}" \
  deterministic="${DETERMINISTIC}" \
  amp="${AMP}" \
  resume="${RESUME}" \
  exist_ok="${EXIST_OK}" \
  project="${PROJECT}" \
  name="${NAME}" 2>&1 | tee "${LOG_PATH}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  write_status "failed" "${TRAIN_STATUS}"
  exit "${TRAIN_STATUS}"
fi

if [[ -d "${WEIGHTS_DIR}" && "${KEEP_EPOCH_CKPTS}" -ge 0 ]]; then
  conda run -n vitpose python "${PROJECT_ROOT}/script/yolo26x_detection_training/prune_yolo_epoch_checkpoints.py" \
    --weights-dir "${WEIGHTS_DIR}" \
    --keep "${KEEP_EPOCH_CKPTS}" | tee -a "${LOG_PATH}"
fi

if [[ "${RUN_TEST}" == "True" || "${RUN_TEST}" == "true" ]]; then
  TEST_MODEL_PATH="${WEIGHTS_DIR}/best.pt"
  if [[ "${TEST_WEIGHTS}" == "last" ]]; then
    TEST_MODEL_PATH="${WEIGHTS_DIR}/last.pt"
  elif [[ ! -f "${TEST_MODEL_PATH}" && -f "${WEIGHTS_DIR}/last.pt" ]]; then
    TEST_MODEL_PATH="${WEIGHTS_DIR}/last.pt"
  fi

  echo "Running YOLO26x detection ${TEST_SPLIT} evaluation..." | tee -a "${LOG_PATH}"

  set +e
  conda run -n vitpose yolo detect val \
    model="${TEST_MODEL_PATH}" \
    data="${DATA}" \
    split="${TEST_SPLIT}" \
    imgsz="${IMGSZ}" \
    batch="${BATCH}" \
    device="${DEVICE}" \
    workers="${WORKERS}" 2>&1 | tee "${TEST_LOG_PATH}"
  TEST_STATUS=${PIPESTATUS[0]}
  set -e

  if [[ "${TEST_STATUS}" -ne 0 ]]; then
    echo "WARNING: ${TEST_SPLIT} evaluation failed with code ${TEST_STATUS}. See: ${TEST_LOG_PATH}" | tee -a "${LOG_PATH}"
  else
    echo "${TEST_SPLIT} evaluation log written to: ${TEST_LOG_PATH}" | tee -a "${LOG_PATH}"
  fi
fi

write_status "finished" "0"
echo "Done. Status: ${STATUS_FILE}"
