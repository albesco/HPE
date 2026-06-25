#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vitpose}"

CONFIG_REL="data/intermediate/Side_above_water_EntireSwim_A/_VitPosePP/generated_configs/swimxyz_vitposepp_huge_A_20260607.py"
CHECKPOINT_REL="runs/vitpose_A_20260607/best_AP_epoch_15.pth"
DATASET_ROOT_REL="data/intermediate/Side_above_water_EntireSwim/_train_canonical"
SPLIT="val"
OUTPUT_DIR_REL="data/output/experiments/vitpose_A_20260609"
OVERLAY_DIR_REL="${OUTPUT_DIR_REL}/overlays_Val"
RESULT_PKL_REL="${OUTPUT_DIR_REL}/keypoints_Val.json"
METRICS_REL="${OUTPUT_DIR_REL}/val_metrics.json"

mkdir -p "${PROJECT_ROOT}/${OUTPUT_DIR_REL}"
rm -rf "${PROJECT_ROOT}/${OVERLAY_DIR_REL}"

ANN_FILE="${PROJECT_ROOT}/${DATASET_ROOT_REL}/annotations/person_keypoints_${SPLIT}.json"
IMG_PREFIX="${PROJECT_ROOT}/${DATASET_ROOT_REL}/${SPLIT}2017/"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/src/vitpose_base/tools/test.py" \
  "${PROJECT_ROOT}/${CONFIG_REL}" \
  "${PROJECT_ROOT}/${CHECKPOINT_REL}" \
  --work-dir "${PROJECT_ROOT}/${OUTPUT_DIR_REL}" \
  --out "${PROJECT_ROOT}/${RESULT_PKL_REL}" \
  --eval mAP \
  --eval-out "${PROJECT_ROOT}/${METRICS_REL}" \
  --device auto \
  --cfg-options \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=1 \
    data.test_dataloader.samples_per_gpu=1 \
    data.test_dataloader.workers_per_gpu=1 \
    data.test.ann_file="${ANN_FILE}" \
    data.test.img_prefix="${IMG_PREFIX}"

conda run -n "${CONDA_ENV}" python "${PROJECT_ROOT}/script/vitpose_generate_test_overlays_from_json.py" \
  --dataset-root "${PROJECT_ROOT}/${DATASET_ROOT_REL}" \
  --split "${SPLIT}" \
  --predictions-json "${PROJECT_ROOT}/${OUTPUT_DIR_REL}/result_keypoints.json" \
  --config "${PROJECT_ROOT}/${CONFIG_REL}" \
  --checkpoint "${PROJECT_ROOT}/${CHECKPOINT_REL}" \
  --output-dir "${PROJECT_ROOT}/${OVERLAY_DIR_REL}" \
  --device auto
