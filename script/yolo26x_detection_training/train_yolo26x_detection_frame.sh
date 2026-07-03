#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET_DIR=""
PRETRAINED_CHECKPOINT="${PROJECT_ROOT}/models/detection/yolo26x.pt"
RUN_NAME="$(date -u +%Y%m%d_%H%M)"
START_EPOCH=1
MAX_EPOCHS=100
LR=0.00100
IMGSZ=768
EARLY_STOP_METRIC="mAP50-95/AP@[.50:.95]"
EARLY_STOP_PATIENCE=3
EARLY_STOP_MIN_DELTA=0.007
KEEP_LAST_N_CHECKPOINTS=10
CHECKPOINT_DIR=""
REPORTS_DIR=""
VAL_METRICS_CSV=""
TEST_OUTPUT_DIR=""
TEST_BBOX_JSON="bbox_Test.json"
TEST_METRICS_JSON="metrics_Test.json"
TEST_OVERLAY_DIR=""
DEVICE=0
BATCH_SIZE=2
NUM_WORKERS=2
USE_TMUX="no"
OVERWRITE="no"
EVALUATE_TEST="yes"
OVERLAY_MAX_IMAGES=0
CONF=0.25

show_help() {
  cat <<'HELP'
Usage:
  bash script/yolo26x_detection_training/train_yolo26x_detection_frame.sh [OPTIONS]

Required:
  --dataset-dir <dir>              Dataset root or _Yolo26x_detection dir/YAML parent.

Options:
  --pretrained-checkpoint <path>   Initial checkpoint. Default: models/detection/yolo26x.pt
  --run-name <name>                Default: UTC YYYYMMDD_HHMM
  --start-epoch <int>              Epoch number represented by the first trained epoch. Default: 1
  --max-epochs <int>               Final epoch number to target. Default: 100
  --lr <float>                     Initial learning rate. Default: 0.00100
  --imgsz <int>                    Image size. Default: 768
  --early-stop-metric <name>       Default: mAP50-95/AP@[.50:.95]
  --early-stop-patience <int>      Default: 3
  --early-stop-min-delta <float>   Default: 0.007
  --keep-last-n-checkpoints <int>  Default: 10
  --checkpoint-dir <dir>           Default: runs/RUN_NAME/checkpoint/
  --reports-dir <dir>              Default: runs/RUN_NAME/reports/
  --val-metrics-csv <path>         Default: runs/RUN_NAME/reports/val_metrics_by_epoch.csv
  --test-output-dir <dir>          Default: data/output/experiments/RUN_NAME/
  --test-bbox-json <name|path>     Default: bbox_Test.json
  --test-metrics-json <name|path>  Default: metrics_Test.json
  --test-overlay-dir <dir>         Default: data/output/experiments/RUN_NAME/overlays_Test/
  --device <id|cpu>                Default: 0
  --batch-size <int>               Default: 2
  --num-workers <int>              Default: 2
  --conf <float>                   Test prediction confidence. Default: 0.25
  --overlay-max-images <int>       0 renders all Test images. Default: 0
  --use-tmux <yes|no>              Default: no
  --evaluate-test <yes|no>         Default: yes
  --overwrite                      Allow replacing existing run/test outputs.
  --help
HELP
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset-dir|--DATASET_DIR) DATASET_DIR="$2"; shift 2 ;;
      --pretrained-checkpoint|--PRETRAINED_CHECKPOINT) PRETRAINED_CHECKPOINT="$2"; shift 2 ;;
      --run-name|--RUN_NAME) RUN_NAME="$2"; shift 2 ;;
      --start-epoch|--START_EPOCH) START_EPOCH="$2"; shift 2 ;;
      --max-epochs|--MAX_EPOCHS) MAX_EPOCHS="$2"; shift 2 ;;
      --lr|--LR) LR="$2"; shift 2 ;;
      --imgsz|--IMGSZ) IMGSZ="$2"; shift 2 ;;
      --early-stop-metric|--EARLY_STOP_METRIC) EARLY_STOP_METRIC="$2"; shift 2 ;;
      --early-stop-patience|--EARLY_STOP_PATIENCE) EARLY_STOP_PATIENCE="$2"; shift 2 ;;
      --early-stop-min-delta|--EARLY_STOP_MIN_DELTA) EARLY_STOP_MIN_DELTA="$2"; shift 2 ;;
      --keep-last-n-checkpoints|--KEEP_LAST_N_CHECKPOINTS) KEEP_LAST_N_CHECKPOINTS="$2"; shift 2 ;;
      --checkpoint-dir|--CHECKPOINT_DIR) CHECKPOINT_DIR="$2"; shift 2 ;;
      --reports-dir|--REPORTS_DIR) REPORTS_DIR="$2"; shift 2 ;;
      --val-metrics-csv|--VAL_METRICS_CSV) VAL_METRICS_CSV="$2"; shift 2 ;;
      --test-output-dir|--TEST_OUTPUT_DIR) TEST_OUTPUT_DIR="$2"; shift 2 ;;
      --test-bbox-json|--TEST_BBOX_JSON) TEST_BBOX_JSON="$2"; shift 2 ;;
      --test-metrics-json|--TEST_METRICS_JSON) TEST_METRICS_JSON="$2"; shift 2 ;;
      --test-overlay-dir|--TEST_OVERLAY_DIR) TEST_OVERLAY_DIR="$2"; shift 2 ;;
      --device|--DEVICE) DEVICE="$2"; shift 2 ;;
      --batch-size|--BATCH_SIZE) BATCH_SIZE="$2"; shift 2 ;;
      --num-workers|--NUM_WORKERS) NUM_WORKERS="$2"; shift 2 ;;
      --conf) CONF="$2"; shift 2 ;;
      --overlay-max-images) OVERLAY_MAX_IMAGES="$2"; shift 2 ;;
      --use-tmux) USE_TMUX="$2"; shift 2 ;;
      --evaluate-test) EVALUATE_TEST="$2"; shift 2 ;;
      --overwrite) OVERWRITE="yes"; shift ;;
      --help|-h) show_help; exit 0 ;;
      *) echo "Unknown option: $1" >&2; show_help; exit 1 ;;
    esac
  done
}

abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${PROJECT_ROOT}/${path}"
  fi
}

resolve_dataset() {
  if [[ -z "${DATASET_DIR}" ]]; then
    echo "ERROR: --dataset-dir is required" >&2
    exit 1
  fi
  DATASET_DIR="$(abs_path "${DATASET_DIR}")"
  if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "ERROR: dataset directory does not exist: ${DATASET_DIR}" >&2
    exit 1
  fi

  if [[ -f "${DATASET_DIR}/swimxyz_side_above_water_yolo26x_detection.yaml" ]]; then
    DATA_YAML="${DATASET_DIR}/swimxyz_side_above_water_yolo26x_detection.yaml"
    DETECTION_ROOT="${DATASET_DIR}"
  elif [[ -f "${DATASET_DIR}/_Yolo26x_detection/swimxyz_side_above_water_yolo26x_detection.yaml" ]]; then
    DATA_YAML="${DATASET_DIR}/_Yolo26x_detection/swimxyz_side_above_water_yolo26x_detection.yaml"
    DETECTION_ROOT="${DATASET_DIR}/_Yolo26x_detection"
  elif [[ "$(basename "${DATASET_DIR}")" == "_train_canonical" && -f "$(dirname "${DATASET_DIR}")/_Yolo26x_detection/swimxyz_side_above_water_yolo26x_detection.yaml" ]]; then
    DETECTION_ROOT="$(dirname "${DATASET_DIR}")/_Yolo26x_detection"
    DATA_YAML="${DETECTION_ROOT}/swimxyz_side_above_water_yolo26x_detection.yaml"
  else
    echo "ERROR: could not resolve YOLO26x detection YAML from dataset dir: ${DATASET_DIR}" >&2
    exit 1
  fi

  for split in train val test; do
    if [[ ! -d "${DETECTION_ROOT}/images/${split}" ]]; then
      echo "ERROR: missing images split directory: ${DETECTION_ROOT}/images/${split}" >&2
      exit 1
    fi
    if [[ ! -d "${DETECTION_ROOT}/labels/${split}" ]]; then
      echo "ERROR: missing labels split directory: ${DETECTION_ROOT}/labels/${split}" >&2
      exit 1
    fi
  done
}

resolve_outputs() {
  PRETRAINED_CHECKPOINT="$(abs_path "${PRETRAINED_CHECKPOINT}")"
  if [[ ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
    echo "ERROR: missing PRETRAINED_CHECKPOINT: ${PRETRAINED_CHECKPOINT}" >&2
    exit 1
  fi
  RUN_DIR="${PROJECT_ROOT}/runs/${RUN_NAME}"
  PROJECT_DIR="$(dirname "${RUN_DIR}")"
  NAME="$(basename "${RUN_DIR}")"
  LOG_DIR="${RUN_DIR}/logs"
  TRAIN_LOG="${LOG_DIR}/train.log"
  TEST_LOG="${LOG_DIR}/test.log"
  STATUS_FILE="${RUN_DIR}/training_status.txt"
  MONITOR_STATUS="${RUN_DIR}/monitor_status.json"

  [[ -z "${CHECKPOINT_DIR}" ]] && CHECKPOINT_DIR="${RUN_DIR}/checkpoint"
  [[ -z "${REPORTS_DIR}" ]] && REPORTS_DIR="${RUN_DIR}/reports"
  [[ -z "${VAL_METRICS_CSV}" ]] && VAL_METRICS_CSV="${REPORTS_DIR}/val_metrics_by_epoch.csv"
  [[ -z "${TEST_OUTPUT_DIR}" ]] && TEST_OUTPUT_DIR="${PROJECT_ROOT}/data/output/experiments/${RUN_NAME}"
  [[ -z "${TEST_OVERLAY_DIR}" ]] && TEST_OVERLAY_DIR="${TEST_OUTPUT_DIR}/overlays_Test"

  CHECKPOINT_DIR="$(abs_path "${CHECKPOINT_DIR}")"
  REPORTS_DIR="$(abs_path "${REPORTS_DIR}")"
  VAL_METRICS_CSV="$(abs_path "${VAL_METRICS_CSV}")"
  TEST_OUTPUT_DIR="$(abs_path "${TEST_OUTPUT_DIR}")"
  TEST_OVERLAY_DIR="$(abs_path "${TEST_OVERLAY_DIR}")"
  if [[ "${TEST_BBOX_JSON}" != /* ]]; then TEST_BBOX_JSON="${TEST_OUTPUT_DIR}/${TEST_BBOX_JSON}"; fi
  if [[ "${TEST_METRICS_JSON}" != /* ]]; then TEST_METRICS_JSON="${TEST_OUTPUT_DIR}/${TEST_METRICS_JSON}"; fi

  if [[ "${START_EPOCH}" -lt 1 ]]; then
    echo "ERROR: START_EPOCH must be >= 1" >&2
    exit 1
  fi
  if [[ "${MAX_EPOCHS}" -lt "${START_EPOCH}" ]]; then
    echo "ERROR: MAX_EPOCHS must be >= START_EPOCH" >&2
    exit 1
  fi
  TRAIN_EPOCHS=$((MAX_EPOCHS - START_EPOCH + 1))

  if [[ "${OVERWRITE}" != "yes" ]]; then
    if [[ -e "${RUN_DIR}" ]]; then
      echo "ERROR: run directory exists, pass --overwrite: ${RUN_DIR}" >&2
      exit 1
    fi
    if [[ -e "${TEST_OUTPUT_DIR}" ]]; then
      echo "ERROR: test output directory exists, pass --overwrite: ${TEST_OUTPUT_DIR}" >&2
      exit 1
    fi
  fi

  if [[ "${OVERWRITE}" == "yes" ]]; then
    rm -rf "${RUN_DIR}" "${TEST_OUTPUT_DIR}"
  fi

  mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${CHECKPOINT_DIR}" "${REPORTS_DIR}" "${TEST_OUTPUT_DIR}" "${TEST_OVERLAY_DIR}"
}

write_status() {
  local phase="$1"
  local exit_code="${2:-}"
  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "phase=${phase}"
    [[ -n "${exit_code}" ]] && echo "exit_code=${exit_code}"
    echo "run_name=${RUN_NAME}"
    echo "run_dir=${RUN_DIR}"
    echo "data_yaml=${DATA_YAML}"
    echo "dataset_dir=${DATASET_DIR}"
    echo "detection_root=${DETECTION_ROOT}"
    echo "pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
    echo "start_epoch=${START_EPOCH}"
    echo "max_epochs=${MAX_EPOCHS}"
    echo "train_epochs=${TRAIN_EPOCHS}"
    echo "lr=${LR}"
    echo "imgsz=${IMGSZ}"
    echo "early_stop_metric=${EARLY_STOP_METRIC}"
    echo "early_stop_patience=${EARLY_STOP_PATIENCE}"
    echo "early_stop_min_delta=${EARLY_STOP_MIN_DELTA}"
    echo "keep_last_n_checkpoints=${KEEP_LAST_N_CHECKPOINTS}"
    echo "checkpoint_dir=${CHECKPOINT_DIR}"
    echo "reports_dir=${REPORTS_DIR}"
    echo "val_metrics_csv=${VAL_METRICS_CSV}"
    echo "test_output_dir=${TEST_OUTPUT_DIR}"
    echo "test_bbox_json=${TEST_BBOX_JSON}"
    echo "test_metrics_json=${TEST_METRICS_JSON}"
    echo "test_overlay_dir=${TEST_OVERLAY_DIR}"
    echo "device=${DEVICE}"
    echo "batch_size=${BATCH_SIZE}"
    echo "num_workers=${NUM_WORKERS}"
    echo "train_log=${TRAIN_LOG}"
    echo "test_log=${TEST_LOG}"
  } > "${STATUS_FILE}"
}

print_resolved_parameters() {
  cat <<EOF
=== YOLO26x Detection Training ===
RUN_NAME: ${RUN_NAME}
DATASET_DIR: ${DATASET_DIR}
DETECTION_ROOT: ${DETECTION_ROOT}
DATA_YAML: ${DATA_YAML}
PRETRAINED_CHECKPOINT: ${PRETRAINED_CHECKPOINT}
START_EPOCH: ${START_EPOCH}
MAX_EPOCHS: ${MAX_EPOCHS}
TRAIN_EPOCHS: ${TRAIN_EPOCHS}
LR: ${LR}
IMGSZ: ${IMGSZ}
EARLY_STOP_METRIC: ${EARLY_STOP_METRIC} -> metrics/mAP50-95(B)
EARLY_STOP_PATIENCE: ${EARLY_STOP_PATIENCE}
EARLY_STOP_MIN_DELTA: ${EARLY_STOP_MIN_DELTA}
KEEP_LAST_N_CHECKPOINTS: ${KEEP_LAST_N_CHECKPOINTS}
RUN_DIR: ${RUN_DIR}
CHECKPOINT_DIR: ${CHECKPOINT_DIR}
REPORTS_DIR: ${REPORTS_DIR}
VAL_METRICS_CSV: ${VAL_METRICS_CSV}
TEST_OUTPUT_DIR: ${TEST_OUTPUT_DIR}
TEST_BBOX_JSON: ${TEST_BBOX_JSON}
TEST_METRICS_JSON: ${TEST_METRICS_JSON}
TEST_OVERLAY_DIR: ${TEST_OVERLAY_DIR}
DEVICE: ${DEVICE}
BATCH_SIZE: ${BATCH_SIZE}
NUM_WORKERS: ${NUM_WORKERS}
EOF
}

sync_checkpoints() {
  local weights_dir="${RUN_DIR}/weights"
  [[ -d "${weights_dir}" ]] || return 0
  find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name '*.pt' -delete
  local file
  for file in "${weights_dir}/best.pt" "${weights_dir}/last.pt"; do
    [[ -f "${file}" ]] && cp -f "${file}" "${CHECKPOINT_DIR}/"
  done
  while IFS= read -r file; do
    cp -f "${file}" "${CHECKPOINT_DIR}/"
  done < <(find "${weights_dir}" -maxdepth 1 -type f -name 'epoch*.pt' | sort -V)
}

run_training() {
  write_status "starting"
  print_resolved_parameters | tee "${TRAIN_LOG}"

  setsid bash -lc "cd '${PROJECT_ROOT}' && conda run -n vitpose yolo detect train \
    model='${PRETRAINED_CHECKPOINT}' \
    data='${DATA_YAML}' \
    epochs='${TRAIN_EPOCHS}' \
    imgsz='${IMGSZ}' \
    batch='${BATCH_SIZE}' \
    device='${DEVICE}' \
    workers='${NUM_WORKERS}' \
    patience=100 \
    lr0='${LR}' \
    save=True \
    save_period=1 \
    val=True \
    split=val \
    plots=True \
    verbose=True \
    seed=0 \
    deterministic=True \
    amp=True \
    resume=False \
    exist_ok=True \
    project='${PROJECT_DIR}' \
    name='${NAME}' >> '${TRAIN_LOG}' 2>&1" &

  local train_pid=$!
  echo "${train_pid}" > "${RUN_DIR}/train.pid"
  write_status "training"

  conda run -n vitpose python "${SCRIPT_DIR}/monitor_yolo_detection_patience.py" \
    --run-dir "${RUN_DIR}" \
    --pid "${train_pid}" \
    --metric "metrics/mAP50-95(B)" \
    --patience "${EARLY_STOP_PATIENCE}" \
    --min-delta "${EARLY_STOP_MIN_DELTA}" \
    --keep-last "${KEEP_LAST_N_CHECKPOINTS}" \
    --poll-seconds 60 \
    --status-json "${MONITOR_STATUS}" 2>&1 | tee -a "${TRAIN_LOG}" &
  local monitor_pid=$!

  set +e
  wait "${train_pid}"
  local train_status=$?
  wait "${monitor_pid}"
  set -e

  if [[ "${train_status}" -ne 0 && "${train_status}" -ne 143 ]]; then
    write_status "failed" "${train_status}"
    return "${train_status}"
  fi

  conda run -n vitpose python "${SCRIPT_DIR}/prune_yolo_epoch_checkpoints.py" \
    --weights-dir "${RUN_DIR}/weights" \
    --keep "${KEEP_LAST_N_CHECKPOINTS}" 2>&1 | tee -a "${TRAIN_LOG}"
  sync_checkpoints
}

export_report() {
  if [[ ! -f "${RUN_DIR}/results.csv" ]]; then
    echo "ERROR: missing results.csv: ${RUN_DIR}/results.csv" >&2
    return 1
  fi
  conda run -n vitpose python "${SCRIPT_DIR}/export_yolo_detection_training_report.py" \
    --results-csv "${RUN_DIR}/results.csv" \
    --out-csv "${VAL_METRICS_CSV}" \
    --output-dir "${REPORTS_DIR}" \
    --start-epoch "${START_EPOCH}" 2>&1 | tee -a "${TRAIN_LOG}"
}

evaluate_test() {
  if [[ "${EVALUATE_TEST}" != "yes" ]]; then
    return 0
  fi
  local best_weights="${RUN_DIR}/weights/best.pt"
  if [[ ! -f "${best_weights}" ]]; then
    echo "ERROR: missing best checkpoint for Test evaluation: ${best_weights}" >&2
    return 1
  fi
  conda run -n vitpose python "${SCRIPT_DIR}/evaluate_yolo_detection_split.py" \
    --model "${best_weights}" \
    --data "${DATA_YAML}" \
    --dataset-root "${DETECTION_ROOT}" \
    --split test \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --workers "${NUM_WORKERS}" \
    --conf "${CONF}" \
    --bbox-json "${TEST_BBOX_JSON}" \
    --metrics-json "${TEST_METRICS_JSON}" \
    --overlay-dir "${TEST_OVERLAY_DIR}" \
    --overlay-max-images "${OVERLAY_MAX_IMAGES}" \
    --overwrite 2>&1 | tee "${TEST_LOG}"
}

run_direct() {
  run_training
  export_report
  evaluate_test
  write_status "finished" "0"
  echo "Done. Status: ${STATUS_FILE}"
}

main() {
  parse_args "$@"
  resolve_dataset
  resolve_outputs

  if [[ "${USE_TMUX}" == "yes" ]]; then
    local session_name="train_yolo26x_detection_${RUN_NAME}"
    if tmux has-session -t "${session_name}" 2>/dev/null; then
      echo "ERROR: tmux session already exists: ${session_name}" >&2
      exit 1
    fi
    local overwrite_arg=""
    [[ "${OVERWRITE}" == "yes" ]] && overwrite_arg="--overwrite"
    tmux new-session -d -s "${session_name}" -x 200 -y 50 \
      "cd '${PROJECT_ROOT}' && bash '${SCRIPT_DIR}/train_yolo26x_detection_frame.sh' --use-tmux no --dataset-dir '${DATASET_DIR}' --pretrained-checkpoint '${PRETRAINED_CHECKPOINT}' --run-name '${RUN_NAME}' --start-epoch '${START_EPOCH}' --max-epochs '${MAX_EPOCHS}' --lr '${LR}' --imgsz '${IMGSZ}' --early-stop-metric '${EARLY_STOP_METRIC}' --early-stop-patience '${EARLY_STOP_PATIENCE}' --early-stop-min-delta '${EARLY_STOP_MIN_DELTA}' --keep-last-n-checkpoints '${KEEP_LAST_N_CHECKPOINTS}' --checkpoint-dir '${CHECKPOINT_DIR}' --reports-dir '${REPORTS_DIR}' --val-metrics-csv '${VAL_METRICS_CSV}' --test-output-dir '${TEST_OUTPUT_DIR}' --test-bbox-json '${TEST_BBOX_JSON}' --test-metrics-json '${TEST_METRICS_JSON}' --test-overlay-dir '${TEST_OVERLAY_DIR}' --device '${DEVICE}' --batch-size '${BATCH_SIZE}' --num-workers '${NUM_WORKERS}' --conf '${CONF}' --overlay-max-images '${OVERLAY_MAX_IMAGES}' --evaluate-test '${EVALUATE_TEST}' ${overwrite_arg}"
    echo "Started tmux session: ${session_name}"
    echo "Attach with: tmux attach -t ${session_name}"
    echo "Run directory: ${RUN_DIR}"
    exit 0
  fi

  run_direct
}

main "$@"
