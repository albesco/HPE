#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash script/vitpose_training/trigger_tmux_on_gpu_free.sh [options] -- <train command...>

Options:
  --session-name NAME       tmux session name to create for the training run
  --poll-seconds N          seconds between GPU checks (default: 60)
  --min-free-mib N          minimum free GPU memory required to launch (default: 8192)
  --gpu-index N             GPU index to monitor (default: 0)
  --log-file PATH           optional watcher log file
  --help                    show this help

Example:
  bash script/vitpose_training/trigger_tmux_on_gpu_free.sh \
    --session-name vitpose_suw_frames_20260701 \
    --min-free-mib 8192 \
    --poll-seconds 60 \
    --log-file logs/vitpose_suw_frames_20260701_trigger.log \
    -- \
    cd /home/albertosco/HPE '&&' bash script/vitpose_training/train_vitpose_frame.sh ...
EOF
}

SESSION_NAME=""
POLL_SECONDS=60
MIN_FREE_MIB=8192
GPU_INDEX=0
LOG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name) SESSION_NAME="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[[ -n "${SESSION_NAME}" ]] || { echo "--session-name is required" >&2; exit 1; }
[[ $# -gt 0 ]] || { echo "Missing training command after --" >&2; exit 1; }

TRAIN_CMD=("$@")

log_line() {
  local line="$1"
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$line"
  if [[ -n "${LOG_FILE}" ]]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$line" >> "$LOG_FILE"
  fi
}

read_gpu_memory() {
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "${GPU_INDEX}" 2>/dev/null
}

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

log_line "Watcher armed for session '${SESSION_NAME}' on GPU ${GPU_INDEX} with minimum free VRAM ${MIN_FREE_MIB} MiB."

while true; do
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    log_line "tmux session '${SESSION_NAME}' already exists; watcher exits without launching."
    exit 0
  fi

  gpu_line="$(read_gpu_memory || true)"
  if [[ -z "${gpu_line}" ]]; then
    log_line "GPU query unavailable; retry in ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
    continue
  fi

  used_mib="$(printf '%s' "${gpu_line}" | cut -d',' -f1 | tr -d ' ')"
  total_mib="$(printf '%s' "${gpu_line}" | cut -d',' -f2 | tr -d ' ')"

  if [[ ! "${used_mib}" =~ ^[0-9]+$ || ! "${total_mib}" =~ ^[0-9]+$ ]]; then
    log_line "Unexpected GPU query output '${gpu_line}'; retry in ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
    continue
  fi

  free_mib=$((total_mib - used_mib))
  log_line "GPU ${GPU_INDEX}: free=${free_mib} MiB used=${used_mib} MiB total=${total_mib} MiB."

  if (( free_mib >= MIN_FREE_MIB )); then
    quoted_cmd="$(printf '%q ' "${TRAIN_CMD[@]}")"
    log_line "Threshold reached; launching tmux session '${SESSION_NAME}'."
    tmux new-session -d -s "${SESSION_NAME}" "${quoted_cmd% }"
    log_line "tmux session '${SESSION_NAME}' created; watcher exits."
    exit 0
  fi

  sleep "${POLL_SECONDS}"
done
