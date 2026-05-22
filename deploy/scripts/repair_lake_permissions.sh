#!/usr/bin/env bash
set -euo pipefail

LAKE_ROOT="${LAKE_ROOT:-/var/lib/quant-lab/lake}"
QUANT_LAB_USER="${QUANT_LAB_USER:-quantlab}"
QUANT_LAB_GROUP="${QUANT_LAB_GROUP:-quantlab}"

if [[ ! -d "${LAKE_ROOT}" ]]; then
  echo "SKIP_REPAIR_LAKE_PERMISSIONS reason=missing_lake_root lake_root=${LAKE_ROOT}"
  exit 0
fi

echo "START_REPAIR_LAKE_PERMISSIONS lake_root=${LAKE_ROOT} owner=${QUANT_LAB_USER}:${QUANT_LAB_GROUP}"
chown -R "${QUANT_LAB_USER}:${QUANT_LAB_GROUP}" "${LAKE_ROOT}"
find "${LAKE_ROOT}" -type d -exec chmod u+rwX,g+rwX,o+rX {} +
find "${LAKE_ROOT}" -type f -exec chmod u+rw,g+rw,o+r {} +
echo "FINISH_REPAIR_LAKE_PERMISSIONS lake_root=${LAKE_ROOT}"
