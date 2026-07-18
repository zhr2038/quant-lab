#!/usr/bin/env bash
set -euo pipefail

queue_root="${QUANT_LAB_RESEARCH_QUEUE_ROOT:-/var/lib/quant-lab/research_queue}"
service_user="${QUANT_LAB_SERVICE_USER:-quantlab}"
research_user="${QUANT_LAB_RESEARCH_USER:-quant-research}"
research_group="${QUANT_LAB_RESEARCH_GROUP:-quant-research}"

if ! getent group "${research_group}" >/dev/null; then
  groupadd --system "${research_group}"
fi

if ! getent passwd "${research_user}" >/dev/null; then
  useradd \
    --system \
    --create-home \
    --home-dir "/var/lib/${research_user}" \
    --shell /bin/bash \
    --gid "${research_group}" \
    "${research_user}"
fi

if ! getent passwd "${service_user}" >/dev/null; then
  echo "quant-lab service user is missing: ${service_user}" >&2
  exit 1
fi

queue_directories=(
  pending running completed failed expired cancelled requests
  requests/pending requests/processing requests/completed requests/failed
  results results/inbox results/imported results/archive results/rejected
  snapshots audit lease status validation
)

install -d -o "${service_user}" -g "${research_group}" -m 2770 "${queue_root}"
for relative_path in "${queue_directories[@]}"; do
  install -d \
    -o "${service_user}" \
    -g "${research_group}" \
    -m 2770 \
    "${queue_root}/${relative_path}"
done

# Existing task directories and metadata may predate the shared-group layout.
# Sealed snapshot descendants intentionally remain read-only (0550/0440).
find "${queue_root}" -xdev -path "${queue_root}/snapshots" -prune -o -type d \
  -exec chown "${service_user}:${research_group}" {} + \
  -exec chmod 2770 {} +
find "${queue_root}" -xdev -path "${queue_root}/snapshots" -prune -o -type f \
  -exec chown "${service_user}:${research_group}" {} + \
  -exec chmod 0660 {} +

# Ownership may need migration on older snapshots, but their immutable modes must
# not be relaxed. -h also avoids following an unexpected symlink.
find "${queue_root}/snapshots" -xdev -mindepth 1 \
  -exec chown -h "${service_user}:${research_group}" {} +

echo "research queue permissions ready: ${queue_root}"
