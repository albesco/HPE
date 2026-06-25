#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vitpose}"
TEST_TOOL="${PROJECT_ROOT}/src/vitpose_base/tools/test.py"
OVERLAY_TOOL="${PROJECT_ROOT}/script/vitpose_training/vitpose_generate_test_overlays_from_json.py"

CHECKPOINT=""
DATASET_DIR=""
BASE_CONFIG=""
MAX_TEST_ITEMS=0
SEED=""
CONF=0
OUTPUT_DIR=""
OVERLAYS_DIR=""
DEVICE="auto"
CROP_SIZE="384x128"
BATCH_SIZE=1
NUM_WORKERS=1
SPLIT="test"

show_help() {
  cat <<'HELP'
Usage:
  bash script/vitpose_prediction/predict_vitpose_frame.sh [OPTIONS]

Required:
  --checkpoint <path>       VitPose++ checkpoint (.pth)
  --dataset-dir <dir>       Main dataset directory containing _train_canonical/ and _VitPosePP/

Outputs:
  --output-dir <dir>        Directory for kp_Test.json, metrics_Test.json, result_keypoints.json, effective_config.py
  --overlays-dir <dir>      Directory for frame/keypoint overlays

Options:
  --base-config <path>      Default: DATASET/_VitPosePP/generated_configs/swimxyz_vitposepp_huge.py
  --max-test-items <int>    Random Test items to evaluate; 0 means all (default: 0)
  --seed <int>              Random seed for sampling (optional; default when sampling: 0)
  --conf <float>            Bbox confidence threshold. If 0, keep only the first bbox per image (default: 0)
  --device <auto|cpu|cuda:N> Default: auto
  --crop-size <WxH>         Test crop size matching the checkpoint (default: 384x128)
  --batch-size <int>        Test batch size (default: 1)
  --num-workers <int>       Test dataloader workers (default: 1)
  --split <name>            Split to evaluate (default: test)
  --help

Default outputs when --output-dir/--overlays-dir are omitted:
  data/output/experiments/vitpose_prediction_<checkpoint-name>/
    result_keypoints.json
    kp_Test.json
    metrics_Test.json
    effective_config.py
    overlays_Test/
HELP
}

abs_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "${PROJECT_ROOT}/${value}"
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --checkpoint) CHECKPOINT="$2"; shift 2 ;;
      --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
      --base-config) BASE_CONFIG="$2"; shift 2 ;;
      --max-test-items) MAX_TEST_ITEMS="$2"; shift 2 ;;
      --seed) SEED="$2"; shift 2 ;;
      --conf) CONF="$2"; shift 2 ;;
      --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
      --overlays-dir) OVERLAYS_DIR="$2"; shift 2 ;;
      --device) DEVICE="$2"; shift 2 ;;
      --crop-size) CROP_SIZE="$2"; shift 2 ;;
      --batch-size) BATCH_SIZE="$2"; shift 2 ;;
      --num-workers) NUM_WORKERS="$2"; shift 2 ;;
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
  CHECKPOINT_ABS="$(abs_path "${CHECKPOINT}")"
  DATASET_ROOT_ABS="$(abs_path "${DATASET_DIR}")"
  if [[ ! -f "${CHECKPOINT_ABS}" ]]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
  fi
  if [[ ! -d "${DATASET_ROOT_ABS}" ]]; then
    echo "ERROR: dataset directory not found: ${DATASET_DIR}" >&2
    exit 1
  fi
  if [[ "$(basename "${DATASET_ROOT_ABS}")" == "_train_canonical" || "$(basename "${DATASET_ROOT_ABS}")" == "_VitPosePP" ]]; then
    echo "ERROR: --dataset-dir must be the main dataset directory, e.g. data/intermediate/SAW_frames" >&2
    echo "       received: ${DATASET_DIR}" >&2
    exit 1
  fi
  CANONICAL_ABS="${DATASET_ROOT_ABS}/_train_canonical"
  VITPOSE_EXPORT_ABS="${DATASET_ROOT_ABS}/_VitPosePP"
  if [[ -z "${BASE_CONFIG}" ]]; then
    BASE_CONFIG_ABS="${VITPOSE_EXPORT_ABS}/generated_configs/swimxyz_vitposepp_huge.py"
  else
    BASE_CONFIG_ABS="$(abs_path "${BASE_CONFIG}")"
  fi
  [[ -d "${CANONICAL_ABS}" ]] || { echo "ERROR: missing canonical dataset: ${CANONICAL_ABS}" >&2; exit 1; }
  [[ -d "${VITPOSE_EXPORT_ABS}" ]] || { echo "ERROR: missing VitPose++ export: ${VITPOSE_EXPORT_ABS}" >&2; exit 1; }
  [[ -f "${BASE_CONFIG_ABS}" ]] || { echo "ERROR: base config not found: ${BASE_CONFIG_ABS}" >&2; exit 1; }
  [[ -f "${TEST_TOOL}" ]] || { echo "ERROR: test tool not found: ${TEST_TOOL}" >&2; exit 1; }
  [[ -f "${OVERLAY_TOOL}" ]] || { echo "ERROR: overlay tool not found: ${OVERLAY_TOOL}" >&2; exit 1; }
  if [[ ! "${CROP_SIZE}" =~ ^([0-9]+)x([0-9]+)$ ]]; then
    echo "ERROR: invalid --crop-size '${CROP_SIZE}'. Expected WxH, e.g. 384x128" >&2
    exit 1
  fi
  CROP_W="${BASH_REMATCH[1]}"
  CROP_H="${BASH_REMATCH[2]}"
  HEATMAP_W=$((CROP_W / 4))
  HEATMAP_H=$((CROP_H / 4))
  [[ -f "${CANONICAL_ABS}/annotations/person_keypoints_${SPLIT}.json" ]] || {
    echo "ERROR: missing split annotations: ${CANONICAL_ABS}/annotations/person_keypoints_${SPLIT}.json" >&2
    exit 1
  }
  [[ -d "${CANONICAL_ABS}/${SPLIT}2017" ]] || {
    echo "ERROR: missing split images: ${CANONICAL_ABS}/${SPLIT}2017" >&2
    exit 1
  }
}

resolve_outputs() {
  local checkpoint_name
  checkpoint_name="$(basename "${CHECKPOINT_ABS%.*}")"

  if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/data/output/experiments/vitpose_prediction_${checkpoint_name}"
  fi
  if [[ -z "${OVERLAYS_DIR}" ]]; then
    OVERLAYS_DIR="${OUTPUT_DIR}/overlays_Test"
  fi

  OUTPUT_DIR_ABS="$(mkdir -p "$(abs_path "${OUTPUT_DIR}")" && cd "$(abs_path "${OUTPUT_DIR}")" && pwd)"
  OVERLAYS_DIR_ABS="$(mkdir -p "$(abs_path "${OVERLAYS_DIR}")" && cd "$(abs_path "${OVERLAYS_DIR}")" && pwd)"
  KP_JSON="${OUTPUT_DIR_ABS}/kp_Test.json"
  METRICS_JSON="${OUTPUT_DIR_ABS}/metrics_Test.json"
  RESULT_JSON="${OUTPUT_DIR_ABS}/result_keypoints.json"
  EFFECTIVE_CONFIG="${OUTPUT_DIR_ABS}/effective_config.py"
  LOG_FILE="${OUTPUT_DIR_ABS}/predict.log"
}

prepare_effective_dataset() {
  local sample_seed="${SEED:-0}"
  local sample_root="${OUTPUT_DIR_ABS}/dataset_view_${SPLIT}_n${MAX_TEST_ITEMS}_seed${sample_seed}_conf${CONF}"
  python3 - "${CANONICAL_ABS}" "${sample_root}" "${SPLIT}" "${MAX_TEST_ITEMS}" "${sample_seed}" "${CONF}" <<'PY'
import json
import random
import shutil
import sys
from pathlib import Path

canonical = Path(sys.argv[1]).resolve()
sample_root = Path(sys.argv[2]).resolve()
split = sys.argv[3]
max_items = int(sys.argv[4])
seed = int(sys.argv[5])
conf = float(sys.argv[6])

ann_path = canonical / "annotations" / f"person_keypoints_{split}.json"
image_dir = canonical / f"{split}2017"
payload = json.loads(ann_path.read_text(encoding="utf-8"))
images_by_id = {int(image["id"]): image for image in payload.get("images", [])}
anns_by_image = {}
for ann in payload.get("annotations", []):
    anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

eligible = []
for image_id, image in sorted(images_by_id.items()):
    anns = anns_by_image.get(image_id, [])
    if conf == 0:
        anns = anns[:1]
    else:
        scored = [ann for ann in anns if float(ann.get("score", 1.0)) >= conf]
        anns = scored
    if anns:
        eligible.append((image_id, image, anns))

if max_items > 0 and len(eligible) > max_items:
    eligible = sorted(random.Random(seed).sample(eligible, max_items), key=lambda item: item[0])

if sample_root.exists():
    shutil.rmtree(sample_root)
(sample_root / "annotations").mkdir(parents=True)
(sample_root / f"{split}2017").mkdir(parents=True)

new_images = []
new_annotations = []
for new_image_id, (_old_image_id, image, anns) in enumerate(eligible, start=1):
    source_image = image_dir / image["file_name"]
    if not source_image.is_file():
        raise FileNotFoundError(source_image)
    target_image = sample_root / f"{split}2017" / image["file_name"]
    target_image.symlink_to(source_image)
    image_copy = dict(image)
    image_copy["id"] = new_image_id
    new_images.append(image_copy)
    for ann_index, ann in enumerate(anns, start=1):
        ann_copy = dict(ann)
        ann_copy["id"] = len(new_annotations) + 1
        ann_copy["image_id"] = new_image_id
        new_annotations.append(ann_copy)

subset = dict(payload)
subset["images"] = new_images
subset["annotations"] = new_annotations
(sample_root / "annotations" / f"person_keypoints_{split}.json").write_text(
    json.dumps(subset, indent=2),
    encoding="utf-8",
)
report = {
    "source_canonical": canonical.as_posix(),
    "sample_root": sample_root.as_posix(),
    "split": split,
    "requested_items": max_items,
    "seed": seed,
    "conf": conf,
    "selected_images": len(new_images),
    "selected_annotations": len(new_annotations),
}
(sample_root / "sample_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(sample_root.as_posix())
PY
  EFFECTIVE_CANONICAL_ABS="${sample_root}"
}

write_effective_config() {
  python3 - "${BASE_CONFIG_ABS}" "${EFFECTIVE_CONFIG}" "${CHECKPOINT_ABS}" "${OUTPUT_DIR_ABS}" "${EFFECTIVE_CANONICAL_ABS}" "${SPLIT}" "${BATCH_SIZE}" "${NUM_WORKERS}" "${CROP_W}" "${CROP_H}" "${HEATMAP_W}" "${HEATMAP_H}" <<'PY'
from pathlib import Path
import sys

base_config = Path(sys.argv[1]).resolve()
effective_config = Path(sys.argv[2]).resolve()
checkpoint = Path(sys.argv[3]).resolve()
work_dir = Path(sys.argv[4]).resolve()
canonical = Path(sys.argv[5]).resolve()
split = sys.argv[6]
batch_size = int(sys.argv[7])
num_workers = int(sys.argv[8])
crop_w = int(sys.argv[9])
crop_h = int(sys.argv[10])
heatmap_w = int(sys.argv[11])
heatmap_h = int(sys.argv[12])

text = base_config.read_text(encoding="utf-8").rstrip()
overlay = f"""

load_from = '{checkpoint.as_posix()}'
resume_from = None
work_dir = '{work_dir.as_posix()}'
model = dict(backbone=dict(img_size=({crop_h}, {crop_w})))
dataset_data_cfg = dict(
    image_size=[{crop_w}, {crop_h}],
    heatmap_size=[{heatmap_w}, {heatmap_h}],
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
data['samples_per_gpu'] = {batch_size}
data['workers_per_gpu'] = {num_workers}
data['test_dataloader'] = dict(samples_per_gpu={batch_size}, workers_per_gpu={num_workers})
data['test']['ann_file'] = '{(canonical / "annotations" / f"person_keypoints_{split}.json").as_posix()}'
data['test']['img_prefix'] = '{(canonical / f"{split}2017").as_posix()}/'
data['test']['data_cfg'] = dataset_data_cfg
prediction_metadata = dict(
    source_script='script/vitpose_prediction/predict_vitpose_frame.sh',
    checkpoint='{checkpoint.as_posix()}',
    dataset_root='{canonical.as_posix()}',
    split='{split}',
)
"""
effective_config.write_text(text + "\n" + overlay.lstrip(), encoding="utf-8")
PY
}

run_prediction() {
  rm -f "${KP_JSON}" "${METRICS_JSON}" "${RESULT_JSON}" "${LOG_FILE}"
  rm -rf "${OVERLAYS_DIR_ABS}"
  mkdir -p "${OVERLAYS_DIR_ABS}"

  echo "=== VitPose++ Prediction ===" | tee "${LOG_FILE}"
  echo "Checkpoint:      ${CHECKPOINT_ABS}" | tee -a "${LOG_FILE}"
  echo "Dataset dir:     ${DATASET_ROOT_ABS}" | tee -a "${LOG_FILE}"
  echo "Canonical dir:   ${CANONICAL_ABS}" | tee -a "${LOG_FILE}"
  echo "Effective data:  ${EFFECTIVE_CANONICAL_ABS}" | tee -a "${LOG_FILE}"
  echo "Base config:     ${BASE_CONFIG_ABS}" | tee -a "${LOG_FILE}"
  echo "Effective config:${EFFECTIVE_CONFIG}" | tee -a "${LOG_FILE}"
  echo "Split:           ${SPLIT}" | tee -a "${LOG_FILE}"
  echo "Max items:       ${MAX_TEST_ITEMS}" | tee -a "${LOG_FILE}"
  echo "Seed:            ${SEED:-0}" | tee -a "${LOG_FILE}"
  echo "Confidence:      ${CONF}" | tee -a "${LOG_FILE}"
  echo "Crop size:       ${CROP_SIZE}" | tee -a "${LOG_FILE}"
  echo "KP JSON:         ${KP_JSON}" | tee -a "${LOG_FILE}"
  echo "Metrics JSON:    ${METRICS_JSON}" | tee -a "${LOG_FILE}"
  echo "Overlays dir:    ${OVERLAYS_DIR_ABS}" | tee -a "${LOG_FILE}"

  conda run -n "${CONDA_ENV}" python "${TEST_TOOL}" \
    "${EFFECTIVE_CONFIG}" \
    "${CHECKPOINT_ABS}" \
    --work-dir "${OUTPUT_DIR_ABS}" \
    --eval mAP \
    --eval-out "${METRICS_JSON}" \
    --device "${DEVICE}" 2>&1 | tee -a "${LOG_FILE}"

  if [[ ! -f "${RESULT_JSON}" ]]; then
    echo "ERROR: expected result JSON was not written: ${RESULT_JSON}" >&2
    exit 1
  fi
  cp "${RESULT_JSON}" "${KP_JSON}"

  conda run -n "${CONDA_ENV}" python "${OVERLAY_TOOL}" \
    --dataset-root "${EFFECTIVE_CANONICAL_ABS}" \
    --split "${SPLIT}" \
    --predictions-json "${KP_JSON}" \
    --config "${EFFECTIVE_CONFIG}" \
    --checkpoint "${CHECKPOINT_ABS}" \
    --output-dir "${OVERLAYS_DIR_ABS}" \
    --device "${DEVICE}" 2>&1 | tee -a "${LOG_FILE}"
}

main() {
  parse_args "$@"
  require_inputs
  resolve_outputs
  prepare_effective_dataset
  write_effective_config
  run_prediction

  echo "Prediction outputs: ${OUTPUT_DIR_ABS}"
  echo "Overlay outputs:    ${OVERLAYS_DIR_ABS}"
}

main "$@"
