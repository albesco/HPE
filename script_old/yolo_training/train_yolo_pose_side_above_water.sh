#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL="${MODEL:-${PROJECT_ROOT}/models/pose/yolo26x-pose.pt}"
DATA="${DATA:-${PROJECT_ROOT}/data/intermediate/Side_above_water/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml}"
PROJECT="${PROJECT:-${PROJECT_ROOT}/runs/yolo26x_pose_side_above_water}"
NAME="${NAME:-yolo26x_pose_coco17}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-1280}"
BATCH="${BATCH:-1}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-2}"
PATIENCE="${PATIENCE:-30}"
SAVE_PERIOD="${SAVE_PERIOD:-1}"
KEEP_EPOCH_CKPTS="${KEEP_EPOCH_CKPTS:-3}"
SEED="${SEED:-0}"
DETERMINISTIC="${DETERMINISTIC:-True}"
AMP="${AMP:-True}"
RESUME="${RESUME:-False}"
EXIST_OK="${EXIST_OK:-True}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
RUN_DIR="${PROJECT}/${NAME}"
WEIGHTS_DIR="${RUN_DIR}/weights"
STATUS_FILE="${STATUS_FILE:-${RUN_DIR}/training_status.txt}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/yolo26x_pose_side_above_water_${TIMESTAMP}.log}"

RUN_TEST="${RUN_TEST:-True}"
TEST_SPLIT="${TEST_SPLIT:-test}"
TEST_WEIGHTS="${TEST_WEIGHTS:-best}"
TEST_CSV="${TEST_CSV:-${RUN_DIR}/reports/test_metrics.csv}"
TEST_LOG_PATH="${TEST_LOG_PATH:-${LOG_DIR}/yolo26x_pose_test_eval_${TIMESTAMP}.log}"

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
    echo "model=${MODEL}"
    echo "data=${DATA}"
    echo "epochs=${EPOCHS}"
    echo "imgsz=${IMGSZ}"
    echo "batch=${BATCH}"
    echo "save_period=${SAVE_PERIOD}"
    echo "keep_epoch_ckpts=${KEEP_EPOCH_CKPTS}"
    echo "run_test=${RUN_TEST}"
    echo "test_split=${TEST_SPLIT}"
    echo "test_weights=${TEST_WEIGHTS}"
    echo "test_csv=${TEST_CSV}"
    echo "test_log_path=${TEST_LOG_PATH}"
  } > "${STATUS_FILE}"
}

if [[ ! -f "${DATA}" ]]; then
  echo "Missing YOLO26x-pose data YAML: ${DATA}" >&2
  echo "Prepare it first with: conda run -n vitpose python script/yolo_training/prepare_yolo_pose_dataset.py --overwrite" >&2
  exit 1
fi

if [[ "${MODEL}" = /* && ! -f "${MODEL}" ]]; then
  echo "Missing YOLO26x-pose model weights: ${MODEL}" >&2
  echo "Place yolo26x-pose.pt there, or set MODEL=yolo26x-pose.pt to let Ultralytics resolve it." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${RUN_DIR}"
write_status "starting"

echo "Logging YOLO26x-pose training to: ${LOG_PATH}"
set +e
conda run -n vitpose yolo pose train \
  model="${MODEL}" \
  data="${DATA}" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  workers="${WORKERS}" \
  patience="${PATIENCE}" \
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
  conda run -n vitpose python "${PROJECT_ROOT}/script/yolo_training/prune_yolo_epoch_checkpoints.py" \
    --weights-dir "${WEIGHTS_DIR}" \
    --keep "${KEEP_EPOCH_CKPTS}" | tee -a "${LOG_PATH}"
fi

if [[ "${RUN_TEST}" == "True" || "${RUN_TEST}" == "true" ]]; then
  mkdir -p "${RUN_DIR}/reports"

  TEST_MODEL_PATH="${WEIGHTS_DIR}/best.pt"
  if [[ "${TEST_WEIGHTS}" == "last" ]]; then
    TEST_MODEL_PATH="${WEIGHTS_DIR}/last.pt"
  elif [[ ! -f "${TEST_MODEL_PATH}" && -f "${WEIGHTS_DIR}/last.pt" ]]; then
    TEST_MODEL_PATH="${WEIGHTS_DIR}/last.pt"
  fi

  echo "Running YOLO26x-pose ${TEST_SPLIT} evaluation (Pose/Box mAP50 and mAP50-95)..." | tee -a "${LOG_PATH}"

  set +e
  conda run -n vitpose --no-capture-output python "${PROJECT_ROOT}/script/yolo_training/evaluate_yolo_pose_split.py" \
    --model "${TEST_MODEL_PATH}" \
    --data "${DATA}" \
    --split "${TEST_SPLIT}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --device "${DEVICE}" \
    --workers "${WORKERS}" \
    --out-csv "${TEST_CSV}" 2>&1 | tee "${TEST_LOG_PATH}"
  TEST_STATUS=${PIPESTATUS[0]}
  set -e

  if [[ "${TEST_STATUS}" -ne 0 ]]; then
    echo "WARNING: test evaluation failed with code ${TEST_STATUS}. See: ${TEST_LOG_PATH}" | tee -a "${LOG_PATH}"
  else
    echo "Test metrics written to: ${TEST_CSV}" | tee -a "${LOG_PATH}"
  fi
fi

write_status "finished" "0"
echo "Done. Status: ${STATUS_FILE}"
