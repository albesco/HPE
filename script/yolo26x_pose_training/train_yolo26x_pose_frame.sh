#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_NAME="$(date -u +%Y%m%d_%H%M)"
USE_TMUX="yes"
CHECKPOINT="${PROJECT_ROOT}/models/pose/yolo26x-pose.pt"
DATASET_DIR=""
IMGSZ=768
LR=0.001
PATIENCE=8
MIN_EPOCHS=20
MIN_DELTA=0.007
KEEP_LAST=10
CHECKPOINT_DIR=""
REPORTS_DIR=""
TEST_KP_JSON=""
TEST_METRICS_JSON=""
TEST_METRICS_CSV=""
OVERLAYS_DIR=""
EPOCHS=100
BATCH=1
DEVICE=0
WORKERS=2

show_help() {
  cat <<'HELP'
Usage:
  bash script/yolo26x_pose_training/train_yolo26x_pose_frame.sh [OPTIONS]

Options:
  --run-name <name>
  --use-tmux <yes|no>
  --checkpoint <path>
  --dataset-dir <dir>
  --imgsz <int>
  --lr <float>
  --patience <int>
  --min-epochs <int>
  --min-delta <float>
  --keep-last <int>
  --checkpoint-dir <dir>
  --reports-dir <dir>
  --test-kp-json <path>
  --test-metrics-json <path>
  --test-metrics-csv <path>
  --overlays-dir <dir>
  --epochs <int>
  --batch <int>
  --device <int|str>
  --workers <int>
  --help
HELP
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-name) RUN_NAME="$2"; shift 2 ;;
      --use-tmux) USE_TMUX="$2"; shift 2 ;;
      --checkpoint) CHECKPOINT="$2"; shift 2 ;;
      --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
      --imgsz) IMGSZ="$2"; shift 2 ;;
      --lr) LR="$2"; shift 2 ;;
      --patience) PATIENCE="$2"; shift 2 ;;
      --min-epochs) MIN_EPOCHS="$2"; shift 2 ;;
      --min-delta) MIN_DELTA="$2"; shift 2 ;;
      --keep-last) KEEP_LAST="$2"; shift 2 ;;
      --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
      --reports-dir) REPORTS_DIR="$2"; shift 2 ;;
      --test-kp-json) TEST_KP_JSON="$2"; shift 2 ;;
      --test-metrics-json) TEST_METRICS_JSON="$2"; shift 2 ;;
      --test-metrics-csv) TEST_METRICS_CSV="$2"; shift 2 ;;
      --overlays-dir) OVERLAYS_DIR="$2"; shift 2 ;;
      --epochs) EPOCHS="$2"; shift 2 ;;
      --batch) BATCH="$2"; shift 2 ;;
      --device) DEVICE="$2"; shift 2 ;;
      --workers) WORKERS="$2"; shift 2 ;;
      --help|-h) show_help; exit 0 ;;
      *) echo "Unknown option: $1" >&2; show_help; exit 1 ;;
    esac
  done
}

normalize_run_name() {
  if [[ "${RUN_NAME}" != yolo26x-pose_* ]]; then
    RUN_NAME="yolo26x-pose_${RUN_NAME}"
  fi
}

resolve_dataset_yaml() {
  local dataset_abs
  dataset_abs="$(cd "${DATASET_DIR}" && pwd)"

  if [[ -f "${dataset_abs}/swimxyz_side_above_water_yolo26x_pose.yaml" ]]; then
    echo "${dataset_abs}/swimxyz_side_above_water_yolo26x_pose.yaml"
    return 0
  fi

  if [[ -f "${dataset_abs}/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml" ]]; then
    echo "${dataset_abs}/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml"
    return 0
  fi

  if [[ "$(basename "${dataset_abs}")" == "_train_canonical" ]]; then
    local sibling_pose_dir
    sibling_pose_dir="$(cd "${dataset_abs}/.." && pwd)/_Yolo26x_pose"
    if [[ -f "${sibling_pose_dir}/swimxyz_side_above_water_yolo26x_pose.yaml" ]]; then
      echo "${sibling_pose_dir}/swimxyz_side_above_water_yolo26x_pose.yaml"
      return 0
    fi
  fi

  echo "ERROR: could not resolve a YOLO pose dataset yaml from ${DATASET_DIR}" >&2
  return 1
}

resolve_paths() {
  if [[ -z "${DATASET_DIR}" ]]; then
    echo "ERROR: --dataset-dir is required" >&2
    exit 1
  fi

  if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "ERROR: Dataset directory does not exist: ${DATASET_DIR}" >&2
    exit 1
  fi

  RUN_DIR="${PROJECT_ROOT}/runs/${RUN_NAME}"
  LOG_DIR="${RUN_DIR}/logs"
  TRAIN_LOG="${LOG_DIR}/train.log"
  TEST_LOG="${LOG_DIR}/test.log"
  MONITOR_STATUS="${RUN_DIR}/monitor_status.json"
  METRIC_BEST_CHECKPOINT="${RUN_DIR}/weights/best_map50_95_pose.pt"

  if [[ -z "${CHECKPOINT_DIR}" ]]; then
    CHECKPOINT_DIR="${RUN_DIR}/checkpoint"
  fi
  if [[ -z "${REPORTS_DIR}" ]]; then
    REPORTS_DIR="${RUN_DIR}/reports"
  fi

  local test_output_root="${PROJECT_ROOT}/data/output/experiments/${RUN_NAME}"
  if [[ -z "${TEST_KP_JSON}" ]]; then
    TEST_KP_JSON="${test_output_root}/kp_Test.json"
  fi
  if [[ -z "${TEST_METRICS_JSON}" ]]; then
    TEST_METRICS_JSON="${test_output_root}/metrics_Test.json"
  fi
  if [[ -z "${TEST_METRICS_CSV}" ]]; then
    TEST_METRICS_CSV="${test_output_root}/metrics_Test.csv"
  fi
  if [[ -z "${OVERLAYS_DIR}" ]]; then
    OVERLAYS_DIR="${test_output_root}/overlays_Test"
  fi

  mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${CHECKPOINT_DIR}" "${REPORTS_DIR}"
  mkdir -p "$(dirname "${TEST_KP_JSON}")" "$(dirname "${TEST_METRICS_JSON}")" "$(dirname "${TEST_METRICS_CSV}")" "${OVERLAYS_DIR}"

  DATA_YAML="$(resolve_dataset_yaml)"

  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "WARNING: checkpoint not found at ${CHECKPOINT}; Ultralytics may try to download it" >&2
  fi
}

sync_checkpoints() {
  local weights_dir="${RUN_DIR}/weights"
  if [[ ! -d "${weights_dir}" ]]; then
    echo "WARNING: missing weights dir: ${weights_dir}" >&2
    return 1
  fi

  find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name '*.pt' -delete
  local copied=0
  local file
  for file in "${weights_dir}/best_map50_95_pose.pt" "${weights_dir}/best.pt" "${weights_dir}/last.pt"; do
    if [[ -f "${file}" ]]; then
      cp -f "${file}" "${CHECKPOINT_DIR}/"
      copied=1
    fi
  done
  while IFS= read -r file; do
    cp -f "${file}" "${CHECKPOINT_DIR}/"
    copied=1
  done < <(find "${weights_dir}" -maxdepth 1 -type f -name 'epoch*.pt' | sort -V)

  if [[ "${copied}" -eq 0 ]]; then
    echo "WARNING: no checkpoints copied to ${CHECKPOINT_DIR}" >&2
    return 1
  fi
}

run_training() {
  echo "=== YOLO26x Pose Training ==="
  echo "Run Name:      ${RUN_NAME}"
  echo "Dataset Dir:   ${DATASET_DIR}"
  echo "Data YAML:     ${DATA_YAML}"
  echo "Checkpoint:    ${CHECKPOINT}"
  echo "Image Size:    ${IMGSZ}"
  echo "Learning Rate: ${LR}"
  echo "Patience:      ${PATIENCE}"
  echo "Min Epochs:    ${MIN_EPOCHS}"
  echo "Min Delta:     ${MIN_DELTA}"
  echo "Keep Last:     ${KEEP_LAST}"
  echo "Run Dir:       ${RUN_DIR}"
  echo "Train Log:     ${TRAIN_LOG}"

  : > "${TRAIN_LOG}"

  setsid bash -lc "cd '${PROJECT_ROOT}' && conda run -n vitpose yolo pose train \
    model='${CHECKPOINT}' \
    data='${DATA_YAML}' \
    epochs='${EPOCHS}' \
    imgsz='${IMGSZ}' \
    lr0='${LR}' \
    batch='${BATCH}' \
    device='${DEVICE}' \
    workers='${WORKERS}' \
    patience=100 \
    save=True \
    save_period=1 \
    val=True \
    split=val \
    max_det=1 \
    plots=True \
    verbose=True \
    seed=0 \
    deterministic=True \
    amp=True \
    resume=False \
    exist_ok=True \
    optimizer=AdamW \
    project='${PROJECT_ROOT}/runs' \
    name='${RUN_NAME}' >> '${TRAIN_LOG}' 2>&1" &

  local train_pid=$!
  echo "${train_pid}" > "${RUN_DIR}/train.pid"

  conda run -n vitpose python "${SCRIPT_DIR}/monitor_yolo_pose_patience.py" \
    --run-dir "${RUN_DIR}" \
    --pid "${train_pid}" \
    --metric "metrics/mAP50-95(P)" \
    --patience "${PATIENCE}" \
    --min-epochs "${MIN_EPOCHS}" \
    --keep-last "${KEEP_LAST}" \
    --min-delta "${MIN_DELTA}" \
    --poll-seconds 60 \
    --status-json "${MONITOR_STATUS}" \
    --best-checkpoint "${METRIC_BEST_CHECKPOINT}" 2>&1 | tee -a "${TRAIN_LOG}" &

  local monitor_pid=$!

  set +e
  wait "${train_pid}"
  local train_status=$?
  wait "${monitor_pid}"
  set -e

  if [[ "${train_status}" -ne 0 && "${train_status}" -ne 143 ]]; then
    echo "Training failed with exit code ${train_status}" >&2
    return "${train_status}"
  fi
}

export_report() {
  local results_csv="${RUN_DIR}/results.csv"
  if [[ ! -f "${results_csv}" ]]; then
    echo "ERROR: missing results.csv: ${results_csv}" >&2
    return 1
  fi

  conda run -n vitpose python "${SCRIPT_DIR}/export_yolo_pose_training_report.py" \
    --results-csv "${results_csv}" \
    --out-csv "${REPORTS_DIR}/val_metrics_by_epoch.csv" \
    --output-dir "${REPORTS_DIR}" 2>&1 | tee -a "${TRAIN_LOG}"
}

evaluate_test() {
  local best_weights="${METRIC_BEST_CHECKPOINT}"
  if [[ ! -f "${best_weights}" ]]; then
    echo "WARNING: missing metric-best checkpoint: ${best_weights}; falling back to Ultralytics best.pt" >&2
    best_weights="${RUN_DIR}/weights/best.pt"
    if [[ ! -f "${best_weights}" ]]; then
      echo "ERROR: missing best checkpoint: ${best_weights}" >&2
      return 1
    fi
  fi

  rm -f "${TEST_KP_JSON}" "${TEST_METRICS_JSON}" "${TEST_METRICS_CSV}"
  rm -rf "${OVERLAYS_DIR}"

  conda run -n vitpose python "${SCRIPT_DIR}/evaluate_yolo_pose_split.py" \
    --model "${best_weights}" \
    --data "${DATA_YAML}" \
    --split test \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --device "${DEVICE}" \
    --workers "${WORKERS}" \
    --out-csv "${TEST_METRICS_CSV}" \
    --out-metrics-json "${TEST_METRICS_JSON}" \
    --out-keypoints-json "${TEST_KP_JSON}" \
    --overlays-dir "${OVERLAYS_DIR}" \
    --max-detections-per-image 1 \
    --overlay-max-images 0 2>&1 | tee "${TEST_LOG}"
}

run_direct() {
  run_training
  sync_checkpoints || true
  export_report
  evaluate_test

  echo "Run directory:  ${RUN_DIR}"
  echo "Checkpoint dir: ${CHECKPOINT_DIR}"
  echo "Reports dir:    ${REPORTS_DIR}"
  echo "Test metrics:   ${TEST_METRICS_JSON}"
}

main() {
  parse_args "$@"
  normalize_run_name
  resolve_paths

  if [[ "${USE_TMUX}" == "yes" ]]; then
    local session_name="train_yolo26x_pose_${RUN_NAME}"
    if tmux has-session -t "${session_name}" 2>/dev/null; then
      echo "tmux session already exists: ${session_name}" >&2
      exit 1
    fi

    tmux new-session -d -s "${session_name}" -x 200 -y 50 \
      "cd '${PROJECT_ROOT}' && bash script/yolo26x_pose_training/train_yolo26x_pose_frame.sh --use-tmux no --run-name '${RUN_NAME}' --checkpoint '${CHECKPOINT}' --dataset-dir '${DATASET_DIR}' --checkpoint-dir '${CHECKPOINT_DIR}' --reports-dir '${REPORTS_DIR}' --overlays-dir '${OVERLAYS_DIR}' --test-kp-json '${TEST_KP_JSON}' --test-metrics-json '${TEST_METRICS_JSON}' --test-metrics-csv '${TEST_METRICS_CSV}' --patience '${PATIENCE}' --min-epochs '${MIN_EPOCHS}' --min-delta '${MIN_DELTA}' --keep-last '${KEEP_LAST}' --lr '${LR}' --imgsz '${IMGSZ}' --epochs '${EPOCHS}' --batch '${BATCH}' --device '${DEVICE}' --workers '${WORKERS}'"

    echo "Started tmux session: ${session_name}"
    echo "Attach with: tmux attach -t ${session_name}"
    echo "Run directory: ${RUN_DIR}"
    exit 0
  fi

  run_direct
}

main "$@"
