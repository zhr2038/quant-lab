#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/opt/quant-lab}"
DEPLOY_USER="${DEPLOY_USER:-ubuntu}"
SERVICE_GROUP="${SERVICE_GROUP:-quantlab}"

install -d -o "${DEPLOY_USER}" -g "${SERVICE_GROUP}" -m 2750 "${APP_ROOT}"

echo "START_REPAIR_DEPLOY_PERMISSIONS app_root=${APP_ROOT} owner=${DEPLOY_USER}:${SERVICE_GROUP}"
chown -R "${DEPLOY_USER}:${SERVICE_GROUP}" "${APP_ROOT}"
find "${APP_ROOT}" -type d -exec chmod u=rwx,g=rx,o=,g+s {} +
find "${APP_ROOT}" -type f -exec chmod u=rwX,g=rX,o= {} +
echo "FINISH_REPAIR_DEPLOY_PERMISSIONS app_root=${APP_ROOT}"
