#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-vitpose}"
SESSION_NAME="${SESSION_NAME:-yolo_vitpose_A_testB}"
DEVICE="${DEVICE:-cuda:0}"

DATASET_ROOT_REL="data/intermediate/Side_above_water_EntireSwim_B/_train_canonical"
YOLO_MODEL_REL="runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/last.pt"
VITPOSE_CONFIG_REL="data/intermediate/Side_above_water_EntireSwim_A/_VitPosePP/generated_configs/swimxyz_vitposepp_huge_A_20260605.py"
VITPOSE_CKPT_REL="runs/vitpose_A_20260605/best_AP_epoch_22.pth"
OUTPUT_ROOT_REL="data/output/experiments/vitpose_A_20260605"
TMP_OUTPUT_ROOT_REL="${OUTPUT_ROOT_REL}/_pipeline_test_b_runs"
CANON_OVERLAYS_REL="${OUTPUT_ROOT_REL}/overlays_Test_B"
CANON_KEYPOINTS_REL="${OUTPUT_ROOT_REL}/test_B_keypoints.json"
CANON_SUMMARY_REL="${OUTPUT_ROOT_REL}/test_B_metrics_summary.json"
CANON_FAILURES_REL="${OUTPUT_ROOT_REL}/test_B_failures.json"
LOG_REL="logs/${SESSION_NAME}_$(date -u +%Y%m%d_%H%M%S).log"

COUNT_IMAGES_CMD="import json; from pathlib import Path; ann=Path('${PROJECT_ROOT}/${DATASET_ROOT_REL}/annotations/person_keypoints_test.json'); print(len(json.loads(ann.read_text()).get('images', [])))"
OVERLAY_COUNT="$(conda run -n "${CONDA_ENV}" python -c "${COUNT_IMAGES_CMD}" | tail -n 1 | tr -d '\r')"

mkdir -p "${PROJECT_ROOT}/${OUTPUT_ROOT_REL}"

RUN_CMD="cd ${PROJECT_ROOT} && conda run -n ${CONDA_ENV} python script/yolo_training/evaluate_yolo_vitpose_map.py --dataset-root ${DATASET_ROOT_REL} --split test --yolo-model ${YOLO_MODEL_REL} --vitpose-config ${VITPOSE_CONFIG_REL} --vitpose-checkpoint ${VITPOSE_CKPT_REL} --output-root ${TMP_OUTPUT_ROOT_REL} --imgsz 1280 --conf 0.25 --overlay-count ${OVERLAY_COUNT} --device ${DEVICE}"
FINALIZE_CMD="cd ${PROJECT_ROOT} && LATEST_RUN=\$(find ${TMP_OUTPUT_ROOT_REL} -maxdepth 1 -type d -name 'test_*' | sort | tail -n 1) && test -n \"\${LATEST_RUN}\" && cp \"\${LATEST_RUN}/yolo_vitpose_keypoints_results.json\" ${CANON_KEYPOINTS_REL} && cp \"\${LATEST_RUN}/summary.json\" ${CANON_SUMMARY_REL} && cp \"\${LATEST_RUN}/failures.json\" ${CANON_FAILURES_REL} && rm -rf ${CANON_OVERLAYS_REL} && cp -a \"\${LATEST_RUN}/overlays\" ${CANON_OVERLAYS_REL}"

tmux has-session -t "${SESSION_NAME}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
}

tmux new-session -d -s "${SESSION_NAME}" -n eval "bash -lc '${RUN_CMD} && ${FINALIZE_CMD}' 2>&1 | tee ${LOG_REL}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Log: ${PROJECT_ROOT}/${LOG_REL}"
