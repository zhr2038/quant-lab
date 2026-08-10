#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)"
DEPLOYED_COMMIT="$(git -C "${REPOSITORY_ROOT}" rev-parse --verify HEAD)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! "${DEPLOYED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid repository HEAD: ${DEPLOYED_COMMIT}" >&2
  exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "missing ${ENV_FILE}; copy .env.example and configure secrets first" >&2
  exit 1
fi

temporary_env="${ENV_FILE}.tmp.$$"
env_mode="$(stat -c '%a' "${ENV_FILE}")"
env_owner="$(stat -c '%u:%g' "${ENV_FILE}")"
cleanup() {
  rm -f -- "${temporary_env}"
}
trap cleanup EXIT

umask 077
awk -v commit="${DEPLOYED_COMMIT}" '
  BEGIN { replaced = 0 }
  /^BUILD_GIT_COMMIT=/ {
    print "BUILD_GIT_COMMIT=" commit
    replaced = 1
    next
  }
  { print }
  END {
    if (!replaced) {
      print "BUILD_GIT_COMMIT=" commit
    }
  }
' "${ENV_FILE}" > "${temporary_env}"
chmod "${env_mode}" "${temporary_env}"
chown "${env_owner}" "${temporary_env}"
mv -f -- "${temporary_env}" "${ENV_FILE}"

cd -- "${SCRIPT_DIR}"
docker compose build --pull quant-export-worker
docker compose up -d --force-recreate quant-export-worker

if ! IMAGE_COMMIT="$(docker exec quant-export-worker cat /app/BUILD_GIT_COMMIT)"; then
  echo "export worker did not start for provenance verification" >&2
  docker compose stop quant-export-worker
  exit 1
fi
if ! RUNTIME_COMMIT="$(
  docker inspect \
    --format '{{range .Config.Env}}{{println .}}{{end}}' \
    quant-export-worker \
    | awk -F= '$1 == "BUILD_GIT_COMMIT" { print $2 }'
)"; then
  echo "export worker runtime provenance is unavailable" >&2
  docker compose stop quant-export-worker
  exit 1
fi

if [[ "${IMAGE_COMMIT}" != "${DEPLOYED_COMMIT}" || "${RUNTIME_COMMIT}" != "${DEPLOYED_COMMIT}" ]]; then
  echo "export worker provenance verification failed" >&2
  docker compose stop quant-export-worker
  exit 1
fi

echo "quant-export-worker deployed at ${DEPLOYED_COMMIT}"
