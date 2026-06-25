#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/albertosco/HPE"
CONFIG_PATH="$PROJECT_ROOT/data/intermediate/Side_above_water_train_vitpose/generated_configs/vitposepp_huge_checkpoint_0_smoke.py"
TRAIN_SCRIPT="$PROJECT_ROOT/src/vitpose_base/tools/train.py"

conda run -n vitpose python "$TRAIN_SCRIPT" "$CONFIG_PATH" --log-interval 20
