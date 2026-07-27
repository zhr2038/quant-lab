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
  /^NAS_RESEARCH_IMAGE_GIT_COMMIT=/ {
    print "NAS_RESEARCH_IMAGE_GIT_COMMIT=" commit
    replaced = 1
    next
  }
  { print }
  END {
    if (!replaced) {
      print "NAS_RESEARCH_IMAGE_GIT_COMMIT=" commit
    }
  }
' "${ENV_FILE}" > "${temporary_env}"
chmod "${env_mode}" "${temporary_env}"
chown "${env_owner}" "${temporary_env}"
mv -f -- "${temporary_env}" "${ENV_FILE}"

cd -- "${SCRIPT_DIR}"
docker compose build --pull quant-research-worker
docker compose up -d --force-recreate quant-research-worker

if ! IMAGE_COMMIT="$(
  docker exec quant-research-worker cat /app/BUILD_GIT_COMMIT
)"; then
  echo "research worker did not start for provenance verification" >&2
  docker compose stop quant-research-worker
  exit 1
fi
if ! RUNTIME_COMMIT="$(
  docker inspect \
    --format '{{range .Config.Env}}{{println .}}{{end}}' \
    quant-research-worker \
    | awk -F= '$1 == "QUANT_RESEARCH_WORKER_COMMIT" { print $2 }'
)"; then
  echo "research worker runtime provenance is unavailable" >&2
  docker compose stop quant-research-worker
  exit 1
fi
if ! MOUNTED_REPOSITORY_COMMIT="$(
  docker exec quant-research-worker \
    git --git-dir=/run/provenance/repository_git rev-parse --verify HEAD
)"; then
  echo "mounted repository provenance is unavailable" >&2
  docker compose stop quant-research-worker
  exit 1
fi

if [[
  "${IMAGE_COMMIT}" != "${DEPLOYED_COMMIT}"
  || "${RUNTIME_COMMIT}" != "${DEPLOYED_COMMIT}"
  || "${MOUNTED_REPOSITORY_COMMIT}" != "${DEPLOYED_COMMIT}"
]]; then
  echo "research worker provenance verification failed" >&2
  docker compose stop quant-research-worker
  exit 1
fi

echo "quant-research-worker deployed at ${DEPLOYED_COMMIT}"
