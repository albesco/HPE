#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-vitpose_A_20260607}"
LOG_REL="logs/${SESSION_NAME}_$(date -u +%Y%m%d_%H%M%S).log"
CMD="cd ${PROJECT_ROOT} && bash script/train_vitpose_A_20260607.sh 2>&1 | tee ${LOG_REL}"

tmux has-session -t "${SESSION_NAME}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
}

tmux new-session -d -s "${SESSION_NAME}" -n train "${CMD}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Top-level log: ${PROJECT_ROOT}/${LOG_REL}"
