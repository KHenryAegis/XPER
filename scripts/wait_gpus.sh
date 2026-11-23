#!/usr/bin/env bash
# Block until N GPUs are free of other users' processes, then print their indices.
# "Free" = < FREE_MIB used and 0 compute apps, stable for 2 consecutive polls.
# Usage: GPUS=$(scripts/wait_gpus.sh 1) ; CUDA_VISIBLE_DEVICES=$GPUS ...
set -euo pipefail
NEED=${1:-1}
FREE_MIB=${FREE_MIB:-1000}
POLL=${POLL:-120}
MAX_WAIT=${MAX_WAIT:-43200}

# No `head` in the pipeline: with pipefail, an early-closing head sends SIGPIPE
# upstream and set -e kills the script exactly when GPUs become free.
free_list() {
  local busy idx uuid mem out=() pid
  # Get compute apps, filter out zombies (PID not in /proc)
  busy=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader || true)
  busy=$(while IFS=', ' read -r pid uuid; do [ -d "/proc/$pid" ] && echo "$uuid"; done <<<"$busy")
  while IFS=', ' read -r idx uuid mem; do
    if [ "$mem" -lt "$FREE_MIB" ] && ! grep -qF "$uuid" <<<"$busy"; then out+=("$idx"); fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
  [ "${#out[@]}" -ge "$NEED" ] && (IFS=,; echo "${out[*]:0:$NEED}")
  return 0
}

waited=0; prev=""
while :; do
  cur=$(free_list)
  if [ -n "$cur" ] && [ "$(tr ',' '\n' <<<"$cur" | wc -l)" -ge "$NEED" ] && [ "$cur" = "$prev" ]; then
    echo "$cur"; exit 0
  fi
  prev=$cur
  [ "$waited" -ge "$MAX_WAIT" ] && { echo "timeout waiting for $NEED free GPUs" >&2; exit 1; }
  echo "$(date +%H:%M) waiting for $NEED free GPU(s); free now: '${cur}'" >&2
  sleep "$POLL"; waited=$((waited + POLL))
done
