#!/usr/bin/env bash
set -euo pipefail

LAKE_ROOT="${LAKE_ROOT:-/var/lib/quant-lab/lake}"
QUANT_LAB_BASE_DIR="${QUANT_LAB_BASE_DIR:-$(dirname "${LAKE_ROOT}")}"
QUANT_LAB_USER="${QUANT_LAB_USER:-quantlab}"
QUANT_LAB_GROUP="${QUANT_LAB_GROUP:-quantlab}"

install -d -o "${QUANT_LAB_USER}" -g "${QUANT_LAB_GROUP}" -m 2775 "${QUANT_LAB_BASE_DIR}"
install -d -o "${QUANT_LAB_USER}" -g "${QUANT_LAB_GROUP}" -m 2775 "${LAKE_ROOT}"

echo "START_REPAIR_LAKE_PERMISSIONS base_dir=${QUANT_LAB_BASE_DIR} lake_root=${LAKE_ROOT} owner=${QUANT_LAB_USER}:${QUANT_LAB_GROUP}"
chown -R "${QUANT_LAB_USER}:${QUANT_LAB_GROUP}" "${LAKE_ROOT}"
find "${LAKE_ROOT}" -type d -exec chmod u+rwX,g+rwX,o+rX,g+s {} +
find "${LAKE_ROOT}" -type f -exec chmod u+rw,g+rw,o+r {} +
echo "FINISH_REPAIR_LAKE_PERMISSIONS lake_root=${LAKE_ROOT}"
