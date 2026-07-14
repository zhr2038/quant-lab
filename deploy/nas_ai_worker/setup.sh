#!/bin/sh
set -eu

cd "$(dirname "$0")"
mkdir -p data secrets
[ -f .env ] || cp .env.example .env

cat <<'MSG'
NAS worker directories are ready.

Before starting:
1. Rotate the cliproxy bearer token that was exposed in chat.
2. Edit deploy/nas_ai_worker/.env.
3. Put the dedicated cloud SSH key in secrets/id_ed25519.
4. Put the cloud host key in secrets/known_hosts.
5. Make the private key readable only by container UID 10001:
     chown 10001:10001 secrets/id_ed25519
     chmod 600 secrets/id_ed25519
     chmod 644 secrets/known_hosts
6. Run:
     docker compose build --pull
     docker compose up -d
MSG
