#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-yolo26x_pose_A_20260605}"

tmux new-session -d -s "${SESSION}" \
  "cd '${PROJECT_ROOT}' && bash script/yolo_training/train_yolo26x_pose_A_20260605.sh"

echo "Started tmux session: ${SESSION}"
echo "Attach with: tmux attach -t ${SESSION}"
