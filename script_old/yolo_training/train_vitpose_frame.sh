#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_SCRIPT_REL="script/train_vitpose_SAW_frames_EntireSwim_20260612.sh"
CONDA_ENV="${CONDA_ENV:-vitpose}"

usage() {
  cat <<'EOF'
Usage:
  script/yolo_training/train_vitpose_frame.sh --dataset-dir PATH --pretrained-checkpoint PATH --max-epochs N [options]

Required:
  --dataset-dir PATH              Canonical dataset root with train/val/test and annotations
  --pretrained-checkpoint PATH    Initial checkpoint path
  --max-epochs N                  Max training epochs

Options:
  --run-name NAME                 Run name; default: UTC timestamp YYYYMMDD_HHMM
  --start-epoch N                 Default: 1
  --lr FLOAT                      Default: 0.00100
  --crop-size WxH                 Default: 384x128
  --early-stop-metric NAME        Default: AP
  --early-stop-patience N         Default: 3
  --early-stop-min-delta FLOAT    Default: 0.007
  --keep-last-n-checkpoints N     Default: 10
  --checkpoint-dir PATH           Default: runs/RUN_NAME/checkpoint
  --reports-dir PATH              Default: runs/RUN_NAME/reports
  --val-metrics-csv PATH          Default: runs/RUN_NAME/reports/val_metrics_by_epoch.csv
  --test-output-dir PATH          Default: data/output/experiments/RUN_NAME
  --test-kp-json NAME_OR_PATH     Default: kp_Test.json
  --test-metrics-json NAME_OR_PATH Default: metrics_Test.json
  --test-overlay-dir PATH         Default: data/output/experiments/RUN_NAME/overlays_Test
  --device VALUE                  Default: auto
  --batch-size N                  Optional override
  --num-workers N                 Optional override
  --overwrite                     Allow reuse of non-empty output dirs/files
  --help                          Show this help

Notes:
  - Reuses the pipeline/conventions of script/train_vitpose_SAW_frames_EntireSwim_20260612.sh
  - START_EPOCH is recorded in the generated config metadata; training starts from PRETRAINED_CHECKPOINT
EOF
}

abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "${PROJECT_ROOT}/${path}"
  fi
}

is_nonempty_dir() {
  local dir="$1"
  [[ -d "$dir" ]] && [[ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || { echo "Missing ${label}: $path" >&2; exit 1; }
}

require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || { echo "Missing ${label}: $path" >&2; exit 1; }
}

resolve_metric_name() {
  local raw="${1,,}"
  case "$raw" in
    ap|map50-95|mAP50-95|ap@[.50:.95]|ap50-95|mAP)
      printf 'AP\n'
      ;;
    *)
      echo "Unsupported early-stop metric '${1}'. Use AP / mAP50-95 / AP@[.50:.95]." >&2
      exit 1
      ;;
  esac
}

RUN_NAME="$(date -u +%Y%m%d_%H%M)"
START_EPOCH=1
LR="0.00100"
CROP_SIZE="384x128"
EARLY_STOP_METRIC_RAW="AP"
EARLY_STOP_PATIENCE=3
EARLY_STOP_MIN_DELTA="0.007"
KEEP_LAST_N_CHECKPOINTS=10
DEVICE="auto"
BATCH_SIZE=""
NUM_WORKERS=""
OVERWRITE=0
DATASET_DIR=""
PRETRAINED_CHECKPOINT=""
MAX_EPOCHS=""
CHECKPOINT_DIR=""
REPORTS_DIR=""
VAL_METRICS_CSV=""
TEST_OUTPUT_DIR=""
TEST_KP_JSON="kp_Test.json"
TEST_METRICS_JSON="metrics_Test.json"
TEST_OVERLAY_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --pretrained-checkpoint) PRETRAINED_CHECKPOINT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --start-epoch) START_EPOCH="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --crop-size) CROP_SIZE="$2"; shift 2 ;;
    --early-stop-metric) EARLY_STOP_METRIC_RAW="$2"; shift 2 ;;
    --early-stop-patience) EARLY_STOP_PATIENCE="$2"; shift 2 ;;
    --early-stop-min-delta) EARLY_STOP_MIN_DELTA="$2"; shift 2 ;;
    --keep-last-n-checkpoints) KEEP_LAST_N_CHECKPOINTS="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --reports-dir) REPORTS_DIR="$2"; shift 2 ;;
    --val-metrics-csv) VAL_METRICS_CSV="$2"; shift 2 ;;
    --test-output-dir) TEST_OUTPUT_DIR="$2"; shift 2 ;;
    --test-kp-json) TEST_KP_JSON="$2"; shift 2 ;;
    --test-metrics-json) TEST_METRICS_JSON="$2"; shift 2 ;;
    --test-overlay-dir) TEST_OVERLAY_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

[[ -n "$DATASET_DIR" ]] || { echo "--dataset-dir is required" >&2; usage >&2; exit 1; }
[[ -n "$PRETRAINED_CHECKPOINT" ]] || { echo "--pretrained-checkpoint is required" >&2; usage >&2; exit 1; }
[[ -n "$MAX_EPOCHS" ]] || { echo "--max-epochs is required" >&2; usage >&2; exit 1; }

EARLY_STOP_METRIC="$(resolve_metric_name "$EARLY_STOP_METRIC_RAW")"

if [[ ! "$CROP_SIZE" =~ ^([0-9]+)x([0-9]+)$ ]]; then
  echo "Invalid --crop-size '${CROP_SIZE}'. Expected WxH, e.g. 384x128." >&2
  exit 1
fi
CROP_W="${BASH_REMATCH[1]}"
CROP_H="${BASH_REMATCH[2]}"
HEATMAP_W=$((CROP_W / 4))
HEATMAP_H=$((CROP_H / 4))

DATASET_DIR_ABS="$(abs_path "$DATASET_DIR")"
PRETRAINED_CHECKPOINT_ABS="$(abs_path "$PRETRAINED_CHECKPOINT")"
BASE_SCRIPT_ABS="${PROJECT_ROOT}/${BASE_SCRIPT_REL}"
DATASET_PARENT="$(cd "${DATASET_DIR_ABS}/.." && pwd)"
LABEL_ROOT_ABS="${DATASET_PARENT}/_VitPosePP"
BASE_CONFIG_ABS="${LABEL_ROOT_ABS}/generated_configs/swimxyz_vitposepp_huge.py"

RUN_ROOT_REL="runs/${RUN_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT_REL}/checkpoint}"
REPORTS_DIR="${REPORTS_DIR:-${RUN_ROOT_REL}/reports}"
VAL_METRICS_CSV="${VAL_METRICS_CSV:-${REPORTS_DIR}/val_metrics_by_epoch.csv}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-data/output/experiments/${RUN_NAME}}"
TEST_OVERLAY_DIR="${TEST_OVERLAY_DIR:-${TEST_OUTPUT_DIR}/overlays_Test}"

CHECKPOINT_DIR_ABS="$(abs_path "$CHECKPOINT_DIR")"
REPORTS_DIR_ABS="$(abs_path "$REPORTS_DIR")"
VAL_METRICS_CSV_ABS="$(abs_path "$VAL_METRICS_CSV")"
TEST_OUTPUT_DIR_ABS="$(abs_path "$TEST_OUTPUT_DIR")"
TEST_OVERLAY_DIR_ABS="$(abs_path "$TEST_OVERLAY_DIR")"

if [[ "$TEST_KP_JSON" = /* ]]; then
  TEST_KP_JSON_ABS="$TEST_KP_JSON"
else
  TEST_KP_JSON_ABS="${TEST_OUTPUT_DIR_ABS}/${TEST_KP_JSON}"
fi
if [[ "$TEST_METRICS_JSON" = /* ]]; then
  TEST_METRICS_JSON_ABS="$TEST_METRICS_JSON"
else
  TEST_METRICS_JSON_ABS="${TEST_OUTPUT_DIR_ABS}/${TEST_METRICS_JSON}"
fi

RUN_ROOT_ABS="$(dirname "${CHECKPOINT_DIR_ABS}")"
STATUS_FILE_ABS="${RUN_ROOT_ABS}/training_status.txt"
MONITOR_JSON_ABS="${RUN_ROOT_ABS}/early_stop_status.json"
TRAIN_PID_FILE_ABS="${RUN_ROOT_ABS}/train_pgid.txt"
TRAIN_STDOUT_ABS="${RUN_ROOT_ABS}/train_stdout.log"
GENERATED_CONFIG_ABS="${RUN_ROOT_ABS}/effective_config.py"

require_file "$BASE_SCRIPT_ABS" "base script"
require_dir "$DATASET_DIR_ABS" "dataset directory"
require_file "$PRETRAINED_CHECKPOINT_ABS" "pretrained checkpoint"
require_dir "${DATASET_DIR_ABS}/annotations" "annotations directory"
require_dir "${DATASET_DIR_ABS}/train2017" "train2017 directory"
require_dir "${DATASET_DIR_ABS}/val2017" "val2017 directory"
require_dir "${DATASET_DIR_ABS}/test2017" "test2017 directory"
require_file "${DATASET_DIR_ABS}/annotations/person_keypoints_train.json" "train annotation file"
require_file "${DATASET_DIR_ABS}/annotations/person_keypoints_val.json" "val annotation file"
require_file "${DATASET_DIR_ABS}/annotations/person_keypoints_test.json" "test annotation file"
require_file "$BASE_CONFIG_ABS" "base dataset config"

if [[ "$OVERWRITE" -ne 1 ]]; then
  if is_nonempty_dir "$CHECKPOINT_DIR_ABS"; then
    echo "Checkpoint dir already exists and is not empty: ${CHECKPOINT_DIR_ABS}. Use --overwrite." >&2
    exit 1
  fi
  if is_nonempty_dir "$REPORTS_DIR_ABS"; then
    echo "Reports dir already exists and is not empty: ${REPORTS_DIR_ABS}. Use --overwrite." >&2
    exit 1
  fi
  if is_nonempty_dir "$TEST_OUTPUT_DIR_ABS"; then
    echo "Test output dir already exists and is not empty: ${TEST_OUTPUT_DIR_ABS}. Use --overwrite." >&2
    exit 1
  fi
  if [[ -f "$VAL_METRICS_CSV_ABS" || -f "$TEST_KP_JSON_ABS" || -f "$TEST_METRICS_JSON_ABS" || -f "$GENERATED_CONFIG_ABS" ]]; then
    echo "One or more output files already exist. Use --overwrite." >&2
    exit 1
  fi
fi

mkdir -p "$CHECKPOINT_DIR_ABS" "$REPORTS_DIR_ABS" "$TEST_OUTPUT_DIR_ABS" "$TEST_OVERLAY_DIR_ABS"

BATCH_LITERAL="None"
WORKERS_LITERAL="None"
if [[ -n "$BATCH_SIZE" ]]; then
  BATCH_LITERAL="$BATCH_SIZE"
fi
if [[ -n "$NUM_WORKERS" ]]; then
  WORKERS_LITERAL="$NUM_WORKERS"
fi

export BASE_CONFIG_ABS
export GENERATED_CONFIG_ABS
export PRETRAINED_CHECKPOINT_ABS
export CHECKPOINT_DIR_ABS
export MAX_EPOCHS
export LR
export CROP_W
export CROP_H
export HEATMAP_W
export HEATMAP_H
export DATASET_DIR_ABS
export LABEL_ROOT_ABS
export RUN_NAME
export START_EPOCH
export EARLY_STOP_METRIC
export EARLY_STOP_PATIENCE
export EARLY_STOP_MIN_DELTA
export KEEP_LAST_N_CHECKPOINTS
export BATCH_LITERAL
export WORKERS_LITERAL

python3 <<'PY'
import os
from pathlib import Path

base_config = Path(os.environ["BASE_CONFIG_ABS"])
output_config = Path(os.environ["GENERATED_CONFIG_ABS"])
text = base_config.read_text(encoding="utf-8")

batch_literal = os.environ["BATCH_LITERAL"]
workers_literal = os.environ["WORKERS_LITERAL"]
batch_expr = "None" if batch_literal == "None" else batch_literal
workers_expr = "None" if workers_literal == "None" else workers_literal

overlay = f"""

load_from = {os.environ["PRETRAINED_CHECKPOINT_ABS"]!r}
resume_from = None
work_dir = {os.environ["CHECKPOINT_DIR_ABS"]!r}
total_epochs = {int(os.environ["MAX_EPOCHS"])}
optimizer = dict(lr={float(os.environ["LR"]):.5f})
checkpoint_config = dict(interval=1, max_keep_ckpts={int(os.environ["KEEP_LAST_N_CHECKPOINTS"])}, create_symlink=True)
evaluation = dict(interval=1, metric='mAP', save_best='AP')
model = dict(backbone=dict(img_size=({int(os.environ["CROP_H"])}, {int(os.environ["CROP_W"])})))
dataset_data_cfg = dict(
    image_size=[{int(os.environ["CROP_W"])}, {int(os.environ["CROP_H"])}],
    heatmap_size=[{int(os.environ["HEATMAP_W"])}, {int(os.environ["HEATMAP_H"])}],
    num_output_channels=17,
    num_joints=17,
    dataset_channel=[list(range(17))],
    inference_channel=list(range(17)),
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    max_num_joints=17,
    dataset_idx=0,
)
data = dict(
    samples_per_gpu={batch_expr if batch_expr != "None" else "4"},
    workers_per_gpu={workers_expr if workers_expr != "None" else "2"},
    val_dataloader=dict(samples_per_gpu={batch_expr if batch_expr != "None" else "4"}),
    test_dataloader=dict(samples_per_gpu={batch_expr if batch_expr != "None" else "4"}),
    train=dict(
        ann_file={str(Path(os.environ["LABEL_ROOT_ABS"]) / "annotations" / "person_keypoints_train.json")!r},
        img_prefix={str(Path(os.environ["DATASET_DIR_ABS"]) / "train2017/")!r},
        data_cfg=dataset_data_cfg,
    ),
    val=dict(
        ann_file={str(Path(os.environ["LABEL_ROOT_ABS"]) / "annotations" / "person_keypoints_val.json")!r},
        img_prefix={str(Path(os.environ["DATASET_DIR_ABS"]) / "val2017/")!r},
        data_cfg=dataset_data_cfg,
    ),
    test=dict(
        ann_file={str(Path(os.environ["LABEL_ROOT_ABS"]) / "annotations" / "person_keypoints_test.json")!r},
        img_prefix={str(Path(os.environ["DATASET_DIR_ABS"]) / "test2017/")!r},
        data_cfg=dataset_data_cfg,
    ),
)
pipeline_metadata = dict(
    tag={os.environ["RUN_NAME"]!r},
    source_base_script='script/train_vitpose_SAW_frames_EntireSwim_20260612.sh',
    selected_config=f'lr_{float(os.environ["LR"]):.5f}_crop_{int(os.environ["CROP_W"])}x{int(os.environ["CROP_H"])}',
    bbox_source='gt_dataset_bboxes',
    start_epoch={int(os.environ["START_EPOCH"])},
    early_stop_metric={os.environ["EARLY_STOP_METRIC"]!r},
    early_stop_metric_semantics='COCO keypoint AP@[0.50:0.95]',
    patience={int(os.environ["EARLY_STOP_PATIENCE"])},
    min_delta={float(os.environ["EARLY_STOP_MIN_DELTA"])},
    keep_last_checkpoints={int(os.environ["KEEP_LAST_N_CHECKPOINTS"])},
)
"""

output_config.write_text(text.rstrip() + "\n" + overlay.lstrip(), encoding="utf-8")
PY

TRAIN_LOG_INTERVAL=20
TRAIN_STATUS_INTERVAL=20
TRAIN_CMD="conda run -n '${CONDA_ENV}' python src/vitpose_base/tools/train.py '${GENERATED_CONFIG_ABS}' --work-dir '${CHECKPOINT_DIR_ABS}' --log-interval ${TRAIN_LOG_INTERVAL} --status-file '${STATUS_FILE_ABS}' --status-interval ${TRAIN_STATUS_INTERVAL}"

echo "Starting VitPose++ training"
echo "base_script=${BASE_SCRIPT_ABS}"
echo "dataset_dir=${DATASET_DIR_ABS}"
echo "pretrained_checkpoint=${PRETRAINED_CHECKPOINT_ABS}"
echo "run_name=${RUN_NAME}"
echo "start_epoch=${START_EPOCH}"
echo "max_epochs=${MAX_EPOCHS}"
echo "lr=${LR}"
echo "crop_size=${CROP_SIZE}"
echo "early_stop_metric=${EARLY_STOP_METRIC}"
echo "early_stop_patience=${EARLY_STOP_PATIENCE}"
echo "early_stop_min_delta=${EARLY_STOP_MIN_DELTA}"
echo "keep_last_n_checkpoints=${KEEP_LAST_N_CHECKPOINTS}"
echo "checkpoint_dir=${CHECKPOINT_DIR_ABS}"
echo "reports_dir=${REPORTS_DIR_ABS}"
echo "val_metrics_csv=${VAL_METRICS_CSV_ABS}"
echo "test_output_dir=${TEST_OUTPUT_DIR_ABS}"
echo "test_kp_json=${TEST_KP_JSON_ABS}"
echo "test_metrics_json=${TEST_METRICS_JSON_ABS}"
echo "test_overlay_dir=${TEST_OVERLAY_DIR_ABS}"
echo "device=${DEVICE}"
echo "batch_size=${BATCH_SIZE:-default-from-config}"
echo "num_workers=${NUM_WORKERS:-default-from-config}"
echo "generated_config=${GENERATED_CONFIG_ABS}"

rm -f "$TRAIN_PID_FILE_ABS"
(
  cd "$PROJECT_ROOT"
  setsid bash -lc "cd '${PROJECT_ROOT}'; exec > >(tee '${TRAIN_STDOUT_ABS}') 2>&1; echo \$BASHPID > '${TRAIN_PID_FILE_ABS}'; exec ${TRAIN_CMD}"
) &
TRAIN_WRAPPER_PID=$!
sleep 2
if [[ -f "$TRAIN_PID_FILE_ABS" ]]; then
  TRAIN_PGID="$(tr -d '[:space:]' < "$TRAIN_PID_FILE_ABS")"
else
  TRAIN_PGID="$(ps -o pgid= "${TRAIN_WRAPPER_PID}" | tr -d ' ')"
fi

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/monitor_vitpose_patience.py" \
  --work-dir "${CHECKPOINT_DIR_ABS}" \
  --pid "${TRAIN_PGID}" \
  --metric "${EARLY_STOP_METRIC}" \
  --patience "${EARLY_STOP_PATIENCE}" \
  --keep-last "${KEEP_LAST_N_CHECKPOINTS}" \
  --min-delta "${EARLY_STOP_MIN_DELTA}" \
  --poll-interval 60 \
  --status-file "${STATUS_FILE_ABS}" \
  --output "${MONITOR_JSON_ABS}"

wait "${TRAIN_WRAPPER_PID}" || true

BEST_CKPT="$(find "${CHECKPOINT_DIR_ABS}" -maxdepth 1 -name 'best_*.pth' | sort -V | tail -n 1)"
if [[ -z "${BEST_CKPT}" && -L "${CHECKPOINT_DIR_ABS}/latest.pth" ]]; then
  BEST_CKPT="$(readlink -f "${CHECKPOINT_DIR_ABS}/latest.pth")"
fi
if [[ -z "${BEST_CKPT}" ]]; then
  echo "No checkpoint found in ${CHECKPOINT_DIR_ABS}" >&2
  exit 1
fi

MMPOSE_LOG="$(find "${CHECKPOINT_DIR_ABS}" -maxdepth 1 -type f -name '*.log' ! -name 'train_stdout.log' | sort | tail -n 1)"
if [[ -z "${MMPOSE_LOG}" ]]; then
  MMPOSE_LOG="${TRAIN_STDOUT_ABS}"
fi
[[ -f "${MMPOSE_LOG}" ]] || { echo "No usable log file found in ${CHECKPOINT_DIR_ABS}" >&2; exit 1; }

if [[ "${DEVICE}" == "auto" ]]; then
  DEVICE="$(conda run -n "${CONDA_ENV}" python -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')" | tail -n 1 | tr -d '\r')"
fi

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/export_vitpose_val_metrics.py" \
  --work-dir "${CHECKPOINT_DIR_ABS}" \
  --out-csv "${VAL_METRICS_CSV_ABS}"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/plot_vitpose_training_log.py" \
  --log-file "${MMPOSE_LOG}" \
  --output-dir "${REPORTS_DIR_ABS}" \
  --timestamp "${RUN_NAME}"

TEST_CFG_OPTIONS=(
  "data.samples_per_gpu=1"
  "data.workers_per_gpu=1"
  "data.test_dataloader.samples_per_gpu=1"
  "data.test_dataloader.workers_per_gpu=1"
  "data.test.ann_file=${DATASET_DIR_ABS}/annotations/person_keypoints_test.json"
  "data.test.img_prefix=${DATASET_DIR_ABS}/test2017/"
)

if [[ -n "${BATCH_SIZE}" ]]; then
  TEST_CFG_OPTIONS[0]="data.samples_per_gpu=${BATCH_SIZE}"
  TEST_CFG_OPTIONS[2]="data.test_dataloader.samples_per_gpu=${BATCH_SIZE}"
fi
if [[ -n "${NUM_WORKERS}" ]]; then
  TEST_CFG_OPTIONS[1]="data.workers_per_gpu=${NUM_WORKERS}"
  TEST_CFG_OPTIONS[3]="data.test_dataloader.workers_per_gpu=${NUM_WORKERS}"
fi

if [[ "$OVERWRITE" -eq 1 ]]; then
  rm -f "${TEST_OUTPUT_DIR_ABS}/result_keypoints.json" "${TEST_KP_JSON_ABS}" "${TEST_METRICS_JSON_ABS}"
  rm -rf "${TEST_OVERLAY_DIR_ABS}"
  mkdir -p "${TEST_OVERLAY_DIR_ABS}"
fi

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/src/vitpose_base/tools/test.py" \
  "${GENERATED_CONFIG_ABS}" \
  "${BEST_CKPT}" \
  --work-dir "${TEST_OUTPUT_DIR_ABS}" \
  --eval mAP \
  --eval-out "${TEST_METRICS_JSON_ABS}" \
  --device "${DEVICE}" \
  --cfg-options "${TEST_CFG_OPTIONS[@]}"

require_file "${TEST_OUTPUT_DIR_ABS}/result_keypoints.json" "test result_keypoints.json"
cp "${TEST_OUTPUT_DIR_ABS}/result_keypoints.json" "${TEST_KP_JSON_ABS}"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/vitpose_generate_test_overlays_from_json.py" \
  --dataset-root "${DATASET_DIR_ABS}" \
  --split test \
  --predictions-json "${TEST_KP_JSON_ABS}" \
  --config "${GENERATED_CONFIG_ABS}" \
  --checkpoint "${BEST_CKPT}" \
  --output-dir "${TEST_OVERLAY_DIR_ABS}" \
  --device "${DEVICE}"

echo "Training and Test outputs completed."
