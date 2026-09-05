#!/usr/bin/env bash
set -euo pipefail
ROOT=/volume2/quant-lab/decision
STATE=/volume1/docker/quant-decision
exec 9>"$ROOT/worker.lock"
flock -n 9 || { echo 'DECISION_WORKER_BUSY'; exit 0; }
image="$(cat "$ROOT/image-ref")"
[[ "$image" =~ ^quant-decision:[a-f0-9]{40}$ ]] || { echo 'INVALID_IMAGE_REF' >&2; exit 2; }
mkdir -p "$ROOT/logs" "$STATE"
available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if (( available_kib < 6291456 )); then
  echo 'DECISION_DEFERRED_NAS_MEMORY_RESERVE_BELOW_6_GIB' >&2
  exit 75
fi
cleanup() {
  label="$(docker inspect --format '{{index .Config.Labels "com.quant-lab.component"}}' quant-decision-job 2>/dev/null || true)"
  if [[ "$label" == decision ]]; then docker stop --time 15 quant-decision-job >/dev/null; fi
}
trap cleanup EXIT
# A one-shot container; no daemon, no automatic restart, no Docker socket mount.
timeout --kill-after=20s 12m docker run --rm --name quant-decision-job \
  --label com.quant-lab.component=decision \
  --cpus=3 --cpu-shares=128 --memory=4g --memory-swap=4g --pids-limit=128 \
  --read-only --cap-drop=ALL --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,size=268435456 \
  --mount "type=bind,src=$STATE,dst=/state" \
  --mount "type=bind,src=$ROOT/archive,dst=/archive" \
  --mount "type=bind,src=$ROOT/private,dst=/private,readonly" \
  --mount 'type=bind,src=/volume2/quant-lab/archive/retirement-20260905/qyun2/data/lake,dst=/bootstrap,readonly' \
  "$image"
