#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vitpose}"

usage() {
  cat <<'EOF'
Usage:
  bash script/vitpose_training/train_vitpose_frame.sh --dataset-dir DATASET_ROOT --pretrained-checkpoint CHECKPOINT --max-epochs N [options]

Required:
  --dataset-dir PATH              Dataset root containing _train_canonical/ and _VitPosePP/
  --pretrained-checkpoint PATH    Initial checkpoint loaded through load_from
  --max-epochs N                  Total training epochs

Options:
  --test-dataset-dir PATH         Dataset root for final Test. Default: same as --dataset-dir
  --base-config PATH              Default: DATASET_ROOT/_VitPosePP/generated_configs/swimxyz_vitposepp_huge.py
  --run-name NAME                 Default: UTC timestamp YYYYMMDD_HHMM
  --start-epoch N                 Metadata only. Default: 1
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
  --test-kp-json NAME_OR_PATH     Default: kp_Test.json under --test-output-dir
  --test-metrics-json NAME_OR_PATH Default: metrics_Test.json under --test-output-dir
  --test-overlay-dir PATH         Default: data/output/experiments/RUN_NAME/overlays_Test
  --yolo-detector-checkpoint PATH Checkpoint for final YOLO26x-Detection -> VitPose++ Test
  --yolo-imgsz N                  YOLO detector image size for final Test. Default: 768
  --yolo-conf FLOAT               YOLO detector confidence threshold. Default: 0.25
  --device VALUE                  Default: auto. Use auto, cpu, cuda:0, ...
  --batch-size N                  Optional override for train/val/test dataloaders
  --num-workers N                 Optional override for train/val/test dataloaders
  --run-test yes|no               Default: yes
  --render-overlays yes|no        Default: yes
  --overwrite                     Allow reuse/removal of existing output files/directories
  --help                          Show this help

Dataset layout:
  DATASET_ROOT/
    _train_canonical/{train2017,val2017,test2017,annotations/}
    _VitPosePP/generated_configs/swimxyz_vitposepp_huge.py
EOF
}

abs_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "${PROJECT_ROOT}/${value}"
  fi
}

is_yes() {
  case "${1,,}" in
    yes|true|1|on) return 0 ;;
    no|false|0|off) return 1 ;;
    *) echo "Expected yes/no value, got: $1" >&2; exit 1 ;;
  esac
}

is_nonempty_dir() {
  local dir="$1"
  [[ -d "$dir" ]] && [[ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || { echo "Missing ${label}: ${path}" >&2; exit 1; }
}

require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || { echo "Missing ${label}: ${path}" >&2; exit 1; }
}

reject_canonical_path() {
  local path="$1"
  local label="$2"
  if [[ "$(basename "$path")" == "_train_canonical" ]]; then
    echo "${label} must be the dataset root, not _train_canonical: ${path}" >&2
    echo "Use the parent directory, for example data/intermediate/SAW_frames." >&2
    exit 1
  fi
}

resolve_metric_name() {
  local raw="${1,,}"
  case "$raw" in
    ap|map|map50-95|ap50-95|ap@[.50:.95]|ap@[0.50:0.95]) printf 'AP\n' ;;
    *) echo "Unsupported early-stop metric '${1}'. Use AP / mAP / mAP50-95." >&2; exit 1 ;;
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
RUN_TEST="yes"
RENDER_OVERLAYS="yes"
OVERWRITE=0
DATASET_DIR=""
TEST_DATASET_DIR=""
BASE_CONFIG=""
PRETRAINED_CHECKPOINT=""
MAX_EPOCHS=""
CHECKPOINT_DIR=""
REPORTS_DIR=""
VAL_METRICS_CSV=""
TEST_OUTPUT_DIR=""
TEST_KP_JSON="kp_Test.json"
TEST_METRICS_JSON="metrics_Test.json"
TEST_OVERLAY_DIR=""
YOLO_DETECTOR_CHECKPOINT="runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt"
YOLO_IMGSZ=768
YOLO_CONF="0.25"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --test-dataset-dir) TEST_DATASET_DIR="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
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
    --yolo-detector-checkpoint) YOLO_DETECTOR_CHECKPOINT="$2"; shift 2 ;;
    --yolo-imgsz) YOLO_IMGSZ="$2"; shift 2 ;;
    --yolo-conf) YOLO_CONF="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --run-test) RUN_TEST="$2"; shift 2 ;;
    --render-overlays) RENDER_OVERLAYS="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[[ -n "$DATASET_DIR" ]] || { echo "--dataset-dir is required" >&2; usage >&2; exit 1; }
[[ -n "$PRETRAINED_CHECKPOINT" ]] || { echo "--pretrained-checkpoint is required" >&2; usage >&2; exit 1; }
[[ -n "$MAX_EPOCHS" ]] || { echo "--max-epochs is required" >&2; usage >&2; exit 1; }

TEST_DATASET_DIR="${TEST_DATASET_DIR:-${DATASET_DIR}}"
EARLY_STOP_METRIC="$(resolve_metric_name "$EARLY_STOP_METRIC_RAW")"
is_yes "$RUN_TEST" && RUN_TEST_ENABLED=1 || RUN_TEST_ENABLED=0
is_yes "$RENDER_OVERLAYS" && RENDER_OVERLAYS_ENABLED=1 || RENDER_OVERLAYS_ENABLED=0

if [[ ! "$CROP_SIZE" =~ ^([0-9]+)x([0-9]+)$ ]]; then
  echo "Invalid --crop-size '${CROP_SIZE}'. Expected WxH, e.g. 384x128." >&2
  exit 1
fi
CROP_W="${BASH_REMATCH[1]}"
CROP_H="${BASH_REMATCH[2]}"
HEATMAP_W=$((CROP_W / 4))
HEATMAP_H=$((CROP_H / 4))

DATASET_ROOT_ABS="$(abs_path "$DATASET_DIR")"
TEST_DATASET_ROOT_ABS="$(abs_path "$TEST_DATASET_DIR")"
reject_canonical_path "$DATASET_ROOT_ABS" "--dataset-dir"
reject_canonical_path "$TEST_DATASET_ROOT_ABS" "--test-dataset-dir"

TRAIN_CANONICAL_ABS="${DATASET_ROOT_ABS}/_train_canonical"
TEST_CANONICAL_ABS="${TEST_DATASET_ROOT_ABS}/_train_canonical"
VITPOSE_EXPORT_ABS="${DATASET_ROOT_ABS}/_VitPosePP"
PRETRAINED_CHECKPOINT_ABS="$(abs_path "$PRETRAINED_CHECKPOINT")"
YOLO_DETECTOR_CHECKPOINT_ABS="$(abs_path "$YOLO_DETECTOR_CHECKPOINT")"
if [[ -n "$BASE_CONFIG" ]]; then
  BASE_CONFIG_ABS="$(abs_path "$BASE_CONFIG")"
else
  BASE_CONFIG_ABS="${VITPOSE_EXPORT_ABS}/generated_configs/swimxyz_vitposepp_huge.py"
fi

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

require_dir "$DATASET_ROOT_ABS" "dataset root"
require_dir "$TRAIN_CANONICAL_ABS" "train canonical dataset"
require_dir "$TEST_CANONICAL_ABS" "test canonical dataset"
require_dir "$VITPOSE_EXPORT_ABS" "VitPose++ export directory"
require_file "$BASE_CONFIG_ABS" "base VitPose++ config"
require_file "$PRETRAINED_CHECKPOINT_ABS" "pretrained checkpoint"
if [[ "$RUN_TEST_ENABLED" -eq 1 ]]; then
  require_file "$YOLO_DETECTOR_CHECKPOINT_ABS" "YOLO26x-Detection checkpoint for final Test"
fi
for split in train val test; do
  require_dir "${TRAIN_CANONICAL_ABS}/${split}2017" "${split} images directory under train dataset"
  require_file "${TRAIN_CANONICAL_ABS}/annotations/person_keypoints_${split}.json" "${split} annotations under train dataset"
done
require_dir "${TEST_CANONICAL_ABS}/test2017" "test images directory under test dataset"
require_file "${TEST_CANONICAL_ABS}/annotations/person_keypoints_test.json" "test annotations under test dataset"

if [[ "$OVERWRITE" -ne 1 ]]; then
  if is_nonempty_dir "$CHECKPOINT_DIR_ABS"; then
    echo "Checkpoint dir already exists and is not empty: ${CHECKPOINT_DIR_ABS}. Use --overwrite." >&2
    exit 1
  fi
  if is_nonempty_dir "$REPORTS_DIR_ABS"; then
    echo "Reports dir already exists and is not empty: ${REPORTS_DIR_ABS}. Use --overwrite." >&2
    exit 1
  fi
  if [[ "$RUN_TEST_ENABLED" -eq 1 ]] && is_nonempty_dir "$TEST_OUTPUT_DIR_ABS"; then
    echo "Test output dir already exists and is not empty: ${TEST_OUTPUT_DIR_ABS}. Use --overwrite." >&2
    exit 1
  fi
  if [[ -f "$VAL_METRICS_CSV_ABS" || -f "$GENERATED_CONFIG_ABS" ]]; then
    echo "One or more run output files already exist. Use --overwrite." >&2
    exit 1
  fi
fi

mkdir -p "$RUN_ROOT_ABS" "$CHECKPOINT_DIR_ABS" "$REPORTS_DIR_ABS"
if [[ "$RUN_TEST_ENABLED" -eq 1 ]]; then
  mkdir -p "$TEST_OUTPUT_DIR_ABS"
  [[ "$RENDER_OVERLAYS_ENABLED" -eq 1 ]] && mkdir -p "$TEST_OVERLAY_DIR_ABS"
fi

BATCH_LITERAL="4"
WORKERS_LITERAL="2"
if [[ -n "$BATCH_SIZE" ]]; then BATCH_LITERAL="$BATCH_SIZE"; fi
if [[ -n "$NUM_WORKERS" ]]; then WORKERS_LITERAL="$NUM_WORKERS"; fi

export BASE_CONFIG_ABS GENERATED_CONFIG_ABS PRETRAINED_CHECKPOINT_ABS CHECKPOINT_DIR_ABS MAX_EPOCHS LR
export CROP_W CROP_H HEATMAP_W HEATMAP_H TRAIN_CANONICAL_ABS TEST_CANONICAL_ABS RUN_NAME START_EPOCH
export EARLY_STOP_METRIC EARLY_STOP_PATIENCE EARLY_STOP_MIN_DELTA KEEP_LAST_N_CHECKPOINTS BATCH_LITERAL WORKERS_LITERAL

python3 <<'PY'
import os
from pathlib import Path

base_config = Path(os.environ["BASE_CONFIG_ABS"])
output_config = Path(os.environ["GENERATED_CONFIG_ABS"])
text = base_config.read_text(encoding="utf-8")

def q(value: str) -> str:
    return repr(value)

train_root = Path(os.environ["TRAIN_CANONICAL_ABS"])
test_root = Path(os.environ["TEST_CANONICAL_ABS"])

overlay = f"""

load_from = {q(os.environ["PRETRAINED_CHECKPOINT_ABS"])}
resume_from = None
work_dir = {q(os.environ["CHECKPOINT_DIR_ABS"])}
total_epochs = {int(os.environ["MAX_EPOCHS"])}
optimizer = dict(lr={float(os.environ["LR"]):.5f})
checkpoint_config = dict(interval=1, max_keep_ckpts={int(os.environ["KEEP_LAST_N_CHECKPOINTS"])}, create_symlink=True)
evaluation = dict(interval=1, metric='mAP', save_best='AP')
model = dict(
    backbone=dict(img_size=({int(os.environ["CROP_H"])}, {int(os.environ["CROP_W"])})),
    associate_keypoint_head=[],
)
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
    _delete_=True,
    samples_per_gpu={int(os.environ["BATCH_LITERAL"])},
    workers_per_gpu={int(os.environ["WORKERS_LITERAL"])},
    val_dataloader=dict(samples_per_gpu={int(os.environ["BATCH_LITERAL"])}),
    test_dataloader=dict(samples_per_gpu={int(os.environ["BATCH_LITERAL"])}),
    train=dict(
        type='TopDownCocoDataset',
        ann_file={q(str(train_root / "annotations" / "person_keypoints_train.json"))},
        img_prefix={q(str(train_root / "train2017") + "/")},
        data_cfg=dataset_data_cfg,
        pipeline=train_pipeline,
        dataset_info=coco_dataset_info,
    ),
    val=dict(
        type='TopDownCocoDataset',
        ann_file={q(str(train_root / "annotations" / "person_keypoints_val.json"))},
        img_prefix={q(str(train_root / "val2017") + "/")},
        data_cfg=dataset_data_cfg,
        pipeline=val_pipeline,
        dataset_info=coco_dataset_info,
    ),
    test=dict(
        type='TopDownCocoDataset',
        ann_file={q(str(test_root / "annotations" / "person_keypoints_test.json"))},
        img_prefix={q(str(test_root / "test2017") + "/")},
        data_cfg=dataset_data_cfg,
        pipeline=test_pipeline,
        dataset_info=coco_dataset_info,
    ),
)
pipeline_metadata = dict(
    tag={q(os.environ["RUN_NAME"])},
    source_script='script/vitpose_training/train_vitpose_frame.sh',
    selected_config=f'lr_{float(os.environ["LR"]):.5f}_crop_{int(os.environ["CROP_W"])}x{int(os.environ["CROP_H"])}',
    bbox_source='gt_dataset_bboxes',
    start_epoch={int(os.environ["START_EPOCH"])},
    early_stop_metric={q(os.environ["EARLY_STOP_METRIC"])},
    early_stop_metric_semantics='COCO keypoint AP@[0.50:0.95]',
    patience={int(os.environ["EARLY_STOP_PATIENCE"])},
    min_delta={float(os.environ["EARLY_STOP_MIN_DELTA"])},
    keep_last_checkpoints={int(os.environ["KEEP_LAST_N_CHECKPOINTS"])},
    train_dataset_root={q(str(train_root))},
    test_dataset_root={q(str(test_root))},
)
"""
output_config.write_text(text.rstrip() + "\n" + overlay.lstrip(), encoding="utf-8")
PY

TRAIN_CMD="conda run -n '${CONDA_ENV}' python src/vitpose_base/tools/train.py '${GENERATED_CONFIG_ABS}' --work-dir '${CHECKPOINT_DIR_ABS}' --log-interval 20 --status-file '${STATUS_FILE_ABS}' --status-interval 20"

echo "Starting VitPose++ training"
echo "script=${SCRIPT_DIR}/train_vitpose_frame.sh"
echo "dataset_root=${DATASET_ROOT_ABS}"
echo "train_canonical=${TRAIN_CANONICAL_ABS}"
echo "test_dataset_root=${TEST_DATASET_ROOT_ABS}"
echo "test_canonical=${TEST_CANONICAL_ABS}"
echo "base_config=${BASE_CONFIG_ABS}"
echo "pretrained_checkpoint=${PRETRAINED_CHECKPOINT_ABS}"
echo "run_name=${RUN_NAME}"
echo "max_epochs=${MAX_EPOCHS}"
echo "lr=${LR}"
echo "crop_size=${CROP_SIZE}"
echo "checkpoint_dir=${CHECKPOINT_DIR_ABS}"
echo "reports_dir=${REPORTS_DIR_ABS}"
echo "test_output_dir=${TEST_OUTPUT_DIR_ABS}"
echo "test_pipeline=Yolo26x-Detection -> VitPose++"
echo "yolo_detector_checkpoint=${YOLO_DETECTOR_CHECKPOINT_ABS}"
echo "yolo_imgsz=${YOLO_IMGSZ}"
echo "yolo_conf=${YOLO_CONF}"
echo "device=${DEVICE}"
echo "generated_config=${GENERATED_CONFIG_ABS}"

if [[ "$OVERWRITE" -eq 1 ]]; then
  rm -f "$VAL_METRICS_CSV_ABS"
  if [[ "$RUN_TEST_ENABLED" -eq 1 ]]; then
    rm -f "${TEST_OUTPUT_DIR_ABS}/result_keypoints.json" "$TEST_KP_JSON_ABS" "$TEST_METRICS_JSON_ABS"
    if [[ "$RENDER_OVERLAYS_ENABLED" -eq 1 ]]; then
      rm -rf "$TEST_OVERLAY_DIR_ABS"
      mkdir -p "$TEST_OVERLAY_DIR_ABS"
    fi
  fi
fi

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

conda run -n "$CONDA_ENV" python "${SCRIPT_DIR}/monitor_vitpose_patience.py" \
  --work-dir "$CHECKPOINT_DIR_ABS" \
  --pid "$TRAIN_PGID" \
  --metric "$EARLY_STOP_METRIC" \
  --patience "$EARLY_STOP_PATIENCE" \
  --keep-last "$KEEP_LAST_N_CHECKPOINTS" \
  --min-delta "$EARLY_STOP_MIN_DELTA" \
  --poll-interval 60 \
  --status-file "$STATUS_FILE_ABS" \
  --output "$MONITOR_JSON_ABS"

wait "$TRAIN_WRAPPER_PID" || true

BEST_CKPT="$(find "$CHECKPOINT_DIR_ABS" -maxdepth 1 -name 'best_*.pth' | sort -V | tail -n 1)"
if [[ -z "$BEST_CKPT" && -L "${CHECKPOINT_DIR_ABS}/latest.pth" ]]; then
  LATEST_TARGET="$(readlink -f "${CHECKPOINT_DIR_ABS}/latest.pth" || true)"
  if [[ -n "$LATEST_TARGET" && -f "$LATEST_TARGET" ]]; then
    BEST_CKPT="$LATEST_TARGET"
  fi
fi
if [[ -z "$BEST_CKPT" ]]; then
  echo "No checkpoint found in ${CHECKPOINT_DIR_ABS}" >&2
  exit 1
fi

MMPOSE_LOG="$(find "$CHECKPOINT_DIR_ABS" -maxdepth 1 -type f -name '*.log' ! -name 'train_stdout.log' | sort | tail -n 1)"
if [[ -z "$MMPOSE_LOG" ]]; then
  MMPOSE_LOG="$TRAIN_STDOUT_ABS"
fi
require_file "$MMPOSE_LOG" "MMPose log"

if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(conda run -n "$CONDA_ENV" python -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')" | tail -n 1 | tr -d '\r')"
fi

echo "Best checkpoint: ${BEST_CKPT}"

conda run -n "$CONDA_ENV" python "${SCRIPT_DIR}/export_vitpose_val_metrics.py" \
  --work-dir "$CHECKPOINT_DIR_ABS" \
  --out-csv "$VAL_METRICS_CSV_ABS"

conda run -n "$CONDA_ENV" python "${SCRIPT_DIR}/plot_vitpose_training_log.py" \
  --log-file "$MMPOSE_LOG" \
  --output-dir "$REPORTS_DIR_ABS" \
  --timestamp "$RUN_NAME"

if [[ "$RUN_TEST_ENABLED" -ne 1 ]]; then
  echo "Training completed. Final Test skipped (--run-test no)."
  exit 0
fi

OVERLAY_COUNT=0
if [[ "$RENDER_OVERLAYS_ENABLED" -eq 1 ]]; then
  OVERLAY_COUNT=-1
fi

conda run -n "$CONDA_ENV" python "${PROJECT_ROOT}/script/yolo26x_detection_prediction/evaluate_yolo_vitpose_map.py" \
  --dataset-root "$TEST_CANONICAL_ABS" \
  --split test \
  --yolo-model "$YOLO_DETECTOR_CHECKPOINT_ABS" \
  --vitpose-config "$GENERATED_CONFIG_ABS" \
  --vitpose-checkpoint "$BEST_CKPT" \
  --output-dir "$TEST_OUTPUT_DIR_ABS" \
  --keypoints-out "$TEST_KP_JSON_ABS" \
  --metrics-out "$TEST_METRICS_JSON_ABS" \
  --summary-out "${TEST_OUTPUT_DIR_ABS}/summary_Test.json" \
  --overlay-dir "$TEST_OVERLAY_DIR_ABS" \
  --imgsz "$YOLO_IMGSZ" \
  --conf "$YOLO_CONF" \
  --overlay-count "$OVERLAY_COUNT" \
  --device "$DEVICE"

require_file "$TEST_KP_JSON_ABS" "YOLO->VitPose test keypoints JSON"
require_file "$TEST_METRICS_JSON_ABS" "YOLO->VitPose test metrics JSON"

echo "Training and Test outputs completed."
