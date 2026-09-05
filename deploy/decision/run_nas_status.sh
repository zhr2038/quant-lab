#!/usr/bin/env bash
set -euo pipefail
ROOT=/volume2/quant-lab/decision
exec 9>"$ROOT/status/collector.lock"
flock -n 9 || exit 0
# Publish explicit collection errors as well. Transport failures leave a dated snapshot.
rc=0
/usr/bin/python3 "$ROOT/status/host_metrics.py" nas --output "$ROOT/status/nas.json" || rc=$?
[[ -s "$ROOT/status/nas.json" ]] || exit 1
timeout --kill-after=3s 20s sftp -q -b - \
  -i "$ROOT/private/id_ed25519" \
  -o "UserKnownHostsFile=$ROOT/private/known_hosts" \
  -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=8 \
  quant-research@qyun2.hrhome.top <<'SFTP'
put /volume2/quant-lab/decision/status/nas.json /var/lib/quant-lab/decision/inbox/.server-status-nas.part
chmod 640 /var/lib/quant-lab/decision/inbox/.server-status-nas.part
rename /var/lib/quant-lab/decision/inbox/.server-status-nas.part /var/lib/quant-lab/decision/inbox/server-status-nas.json
SFTP
exit "$rc"
