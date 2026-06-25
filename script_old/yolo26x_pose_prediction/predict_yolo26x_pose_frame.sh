#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_SCRIPT="${PROJECT_ROOT}/script/yolo26x_pose_training/evaluate_yolo_pose_split.py"

CHECKPOINT=""
DATASET_DIR=""
MAX_TEST_ITEMS=0
SEED=""
CONF=0
MAX_DETECTIONS_PER_IMAGE=1
OUTPUT_DIR=""
OVERLAYS_DIR=""
IMGSZ=768
BATCH=1
DEVICE=0
WORKERS=2
SPLIT="test"

show_help() {
  cat <<'HELP'
Usage:
  bash script/yolo26x_pose_prediction/predict_yolo26x_pose_frame.sh [OPTIONS]

Required:
  --checkpoint <path>       YOLO26x-Pose checkpoint (.pt)
  --dataset-dir <dir>       Main dataset directory containing _Yolo26x_pose/

Outputs:
  --output-dir <dir>        Directory for kp_Test.json, metrics_Test.json, metrics_Test.csv
  --overlays-dir <dir>      Directory for frame/keypoint overlays

Options:
  --max-test-items <int>    Random Test items to evaluate; 0 means all (default: 0)
  --seed <int>              Random seed for sampling (optional; default when sampling: 0)
  --conf <float>            YOLO confidence threshold; 0 means all predictions (default: 0)
  --max-detections-per-image <int>  Detections to export/draw per image; 0 means all (default: 1)
  --imgsz <int>             YOLO image size (default: 768)
  --batch <int>             YOLO val batch size (default: 1)
  --device <int|str>        YOLO device, e.g. 0 or cpu (default: 0)
  --workers <int>           Data loader workers (default: 2)
  --split <name>            Split to evaluate (default: test)
  --help

Default outputs when --output-dir/--overlays-dir are omitted:
  data/output/experiments/yolo26x-pose_prediction_<checkpoint-name>/
    kp_Test.json
    metrics_Test.json
    metrics_Test.csv
    overlays_Test/
HELP
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --checkpoint) CHECKPOINT="$2"; shift 2 ;;
      --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
      --max-test-items) MAX_TEST_ITEMS="$2"; shift 2 ;;
      --seed) SEED="$2"; shift 2 ;;
      --conf) CONF="$2"; shift 2 ;;
      --max-detections-per-image) MAX_DETECTIONS_PER_IMAGE="$2"; shift 2 ;;
      --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
      --overlays-dir) OVERLAYS_DIR="$2"; shift 2 ;;
      --imgsz) IMGSZ="$2"; shift 2 ;;
      --batch) BATCH="$2"; shift 2 ;;
      --device) DEVICE="$2"; shift 2 ;;
      --workers) WORKERS="$2"; shift 2 ;;
      --split) SPLIT="$2"; shift 2 ;;
      --help|-h) show_help; exit 0 ;;
      *) echo "Unknown option: $1" >&2; show_help; exit 1 ;;
    esac
  done
}

require_inputs() {
  if [[ -z "${CHECKPOINT}" ]]; then
    echo "ERROR: --checkpoint is required" >&2
    exit 1
  fi
  if [[ -z "${DATASET_DIR}" ]]; then
    echo "ERROR: --dataset-dir is required" >&2
    exit 1
  fi
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
  fi
  if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "ERROR: evaluator not found: ${EVAL_SCRIPT}" >&2
    exit 1
  fi
}

resolve_dataset_yaml() {
  local input="$1"
  local input_abs
  local yaml_path

  if [[ -f "${input}" ]]; then
    echo "ERROR: --dataset-dir must be the main dataset directory, not a YAML file: ${input}" >&2
    return 1
  fi

  if [[ ! -d "${input}" ]]; then
    echo "ERROR: dataset directory does not exist: ${input}" >&2
    return 1
  fi

  input_abs="$(cd "${input}" && pwd)"

  if [[ "$(basename "${input_abs}")" == "_train_canonical" || "$(basename "${input_abs}")" == "_Yolo26x_pose" ]]; then
    echo "ERROR: --dataset-dir must be the main dataset directory, e.g. data/intermediate/SAW_frames" >&2
    echo "       received: ${input}" >&2
    return 1
  fi

  yaml_path="${input_abs}/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml"
  if [[ -f "${yaml_path}" ]]; then
    echo "${yaml_path}"
    return 0
  fi

  echo "ERROR: could not find YOLO pose YAML under ${input_abs}/_Yolo26x_pose" >&2
  echo "       expected: ${yaml_path}" >&2
  return 1
}

resolve_outputs() {
  local checkpoint_name
  checkpoint_name="$(basename "${CHECKPOINT%.*}")"

  if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/data/output/experiments/yolo26x-pose_prediction_${checkpoint_name}"
  fi
  if [[ -z "${OVERLAYS_DIR}" ]]; then
    OVERLAYS_DIR="${OUTPUT_DIR}/overlays_Test"
  fi

  OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"
  OVERLAYS_DIR="$(mkdir -p "${OVERLAYS_DIR}" && cd "${OVERLAYS_DIR}" && pwd)"
  KP_JSON="${OUTPUT_DIR}/kp_Test.json"
  METRICS_JSON="${OUTPUT_DIR}/metrics_Test.json"
  METRICS_CSV="${OUTPUT_DIR}/metrics_Test.csv"
  LOG_FILE="${OUTPUT_DIR}/predict.log"
}

make_sampled_dataset_if_needed() {
  DATA_YAML="$(resolve_dataset_yaml "${DATASET_DIR}")"
  EFFECTIVE_DATA_YAML="${DATA_YAML}"

  if [[ "${MAX_TEST_ITEMS}" -le 0 ]]; then
    return 0
  fi

  local sample_seed="${SEED:-0}"
  local sample_root="${OUTPUT_DIR}/dataset_view_${SPLIT}_n${MAX_TEST_ITEMS}_seed${sample_seed}"

  conda run -n vitpose python - "${DATA_YAML}" "${sample_root}" "${SPLIT}" "${MAX_TEST_ITEMS}" "${sample_seed}" <<'PY'
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

data_yaml = Path(sys.argv[1]).resolve()
sample_root = Path(sys.argv[2]).resolve()
split = sys.argv[3]
max_items = int(sys.argv[4])
seed = int(sys.argv[5])

cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
base = Path(cfg["path"])
if not base.is_absolute():
    base = (data_yaml.parent / base).resolve()

image_dir = (base / cfg[split]).resolve()
label_dir = Path(str(image_dir).replace("/images/", "/labels/")).resolve()
if not image_dir.is_dir():
    raise FileNotFoundError(image_dir)
if not label_dir.is_dir():
    raise FileNotFoundError(label_dir)

images = sorted(
    p for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and (label_dir / f"{p.stem}.txt").is_file()
)
if max_items > 0 and len(images) > max_items:
    images = sorted(random.Random(seed).sample(images, max_items))

if sample_root.exists():
    shutil.rmtree(sample_root)
(sample_root / "images" / split).mkdir(parents=True)
(sample_root / "labels" / split).mkdir(parents=True)

for image_path in images:
    label_path = label_dir / f"{image_path.stem}.txt"
    image_target = sample_root / "images" / split / image_path.name
    label_target = sample_root / "labels" / split / label_path.name
    image_target.symlink_to(image_path)
    label_target.symlink_to(label_path)

sample_cfg = dict(cfg)
sample_cfg["path"] = sample_root.as_posix()
sample_cfg["train"] = f"images/{split}"
sample_cfg["val"] = f"images/{split}"
sample_cfg["test"] = f"images/{split}"
sample_yaml = sample_root / "yolo_pose_sample.yaml"
sample_yaml.write_text(yaml.safe_dump(sample_cfg, sort_keys=False), encoding="utf-8")

report = {
    "source_yaml": data_yaml.as_posix(),
    "sample_yaml": sample_yaml.as_posix(),
    "split": split,
    "requested_items": max_items,
    "seed": seed,
    "selected_images": len(images),
}
(sample_root / "sample_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(sample_yaml.as_posix())
PY

  EFFECTIVE_DATA_YAML="${sample_root}/yolo_pose_sample.yaml"
}

run_prediction() {
  rm -f "${KP_JSON}" "${METRICS_JSON}" "${METRICS_CSV}" "${LOG_FILE}"
  rm -rf "${OVERLAYS_DIR}"
  mkdir -p "${OVERLAYS_DIR}"

  echo "=== YOLO26x-Pose Prediction ===" | tee "${LOG_FILE}"
  echo "Checkpoint:     ${CHECKPOINT}" | tee -a "${LOG_FILE}"
  echo "Dataset dir:    ${DATASET_DIR}" | tee -a "${LOG_FILE}"
  echo "Dataset YAML:   ${DATA_YAML}" | tee -a "${LOG_FILE}"
  echo "Effective YAML: ${EFFECTIVE_DATA_YAML}" | tee -a "${LOG_FILE}"
  echo "Split:          ${SPLIT}" | tee -a "${LOG_FILE}"
  echo "Max items:      ${MAX_TEST_ITEMS}" | tee -a "${LOG_FILE}"
  echo "Seed:           ${SEED:-0}" | tee -a "${LOG_FILE}"
  echo "Confidence:     ${CONF}" | tee -a "${LOG_FILE}"
  echo "Max detections: ${MAX_DETECTIONS_PER_IMAGE}" | tee -a "${LOG_FILE}"
  echo "KP JSON:        ${KP_JSON}" | tee -a "${LOG_FILE}"
  echo "Metrics JSON:   ${METRICS_JSON}" | tee -a "${LOG_FILE}"
  echo "Metrics CSV:    ${METRICS_CSV}" | tee -a "${LOG_FILE}"
  echo "Overlays dir:   ${OVERLAYS_DIR}" | tee -a "${LOG_FILE}"

  conda run -n vitpose python "${EVAL_SCRIPT}" \
    --model "${CHECKPOINT}" \
    --data "${EFFECTIVE_DATA_YAML}" \
    --split "${SPLIT}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --device "${DEVICE}" \
    --workers "${WORKERS}" \
    --conf "${CONF}" \
    --max-detections-per-image "${MAX_DETECTIONS_PER_IMAGE}" \
    --out-csv "${METRICS_CSV}" \
    --out-metrics-json "${METRICS_JSON}" \
    --out-keypoints-json "${KP_JSON}" \
    --overlays-dir "${OVERLAYS_DIR}" \
    --overlay-max-images 0 \
    --overlay-seed "${SEED:-0}" 2>&1 | tee -a "${LOG_FILE}"
}

main() {
  parse_args "$@"
  require_inputs
  resolve_outputs
  make_sampled_dataset_if_needed
  run_prediction

  echo "Prediction outputs: ${OUTPUT_DIR}"
  echo "Overlay outputs:    ${OVERLAYS_DIR}"
}

main "$@"
