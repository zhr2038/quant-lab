#!/usr/bin/env bash
set -euo pipefail

CLOUD_HOST="${QUANT_ARCHIVE_CLOUD_HOST:-qyun2.hrhome.top}"
CLOUD_PORT="${QUANT_ARCHIVE_CLOUD_PORT:-22}"
CLOUD_USER="${QUANT_ARCHIVE_CLOUD_USER:-quant-research}"
SSH_KEY="${QUANT_ARCHIVE_SSH_KEY:-/volume1/docker/quant-research/secrets/id_ed25519}"
KNOWN_HOSTS="${QUANT_ARCHIVE_KNOWN_HOSTS:-/volume1/docker/quant-research/secrets/known_hosts}"
SOURCE_ROOT="${QUANT_ARCHIVE_SOURCE_ROOT:-/var/lib/quant-lab/archive/v5/bundles}"
DEST_ROOT="${QUANT_ARCHIVE_DEST_ROOT:-/volume1/docker/quant-archive/qyun2/redacted-v5}"
AUDIT_ROOT="${QUANT_ARCHIVE_AUDIT_ROOT:-/volume1/docker/quant-archive/qyun2/audit}"
RETENTION_DAYS="${QUANT_ARCHIVE_RETENTION_DAYS:-45}"
TRANSFER_TIMEOUT_SECONDS="${QUANT_ARCHIVE_TRANSFER_TIMEOUT_SECONDS:-10800}"

case "$DEST_ROOT" in
  /volume1/docker/quant-archive/qyun2/*) ;;
  *) echo "unsafe destination root: $DEST_ROOT" >&2; exit 2 ;;
esac
case "$RETENTION_DAYS" in
  ''|*[!0-9]*) echo "invalid retention days" >&2; exit 2 ;;
esac

mkdir -p "$DEST_ROOT" "$AUDIT_ROOT"
exec 9>"$AUDIT_ROOT/redacted-v5.lock"
flock -n 9 || exit 0

SSH=(
  ssh -i "$SSH_KEY" -p "$CLOUD_PORT" -o BatchMode=yes
  -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3
  -o "UserKnownHostsFile=$KNOWN_HOSTS"
)
RSYNC_SSH="ssh -i $SSH_KEY -p $CLOUD_PORT -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o UserKnownHostsFile=$KNOWN_HOSTS"
REMOTE="$CLOUD_USER@$CLOUD_HOST"
AUDIT_LOG="$AUDIT_ROOT/redacted-v5.jsonl"

today_utc="$("${SSH[@]}" "$REMOTE" date -u +%F)"
mapfile -t source_days < <(
  "${SSH[@]}" "$REMOTE" \
    "find '$SOURCE_ROOT' -mindepth 1 -maxdepth 1 -type d -printf '%f\\n'" \
    | LC_ALL=C sort
)

for day in "${source_days[@]}"; do
  [[ "$day" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || continue
  [[ "$day" < "$today_utc" ]] || continue

  stage="$DEST_ROOT/.${day}.partial"
  final="$DEST_ROOT/$day"
  backup="$DEST_ROOT/.${day}.previous"
  remote_manifest="$(mktemp "$AUDIT_ROOT/.remote-manifest.XXXXXX")"
  local_manifest="$(mktemp "$AUDIT_ROOT/.local-manifest.XXXXXX")"
  trap 'rm -f "$remote_manifest" "$local_manifest"' EXIT

  if [[ -f "$final/.archive_receipt.json" && -f "$final/.archive_manifest.sha256" ]]; then
    rm -f "$remote_manifest" "$local_manifest"
    trap - EXIT
    continue
  fi

  rm -rf -- "$stage"
  mkdir -p "$stage"
  "${SSH[@]}" "$REMOTE" \
    "cd '$SOURCE_ROOT/$day' && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum" \
    >"$remote_manifest"
  timeout "$TRANSFER_TIMEOUT_SECONDS" rsync \
    --archive --partial --delay-updates --delete-delay --safe-links \
    -e "$RSYNC_SSH" "$REMOTE:$SOURCE_ROOT/$day/" "$stage/"
  (
    cd "$stage"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
  ) >"$local_manifest"
  cmp --silent "$remote_manifest" "$local_manifest" || {
    echo "archive checksum mismatch: $day" >&2
    exit 1
  }

  file_count="$(wc -l <"$local_manifest" | tr -d ' ')"
  byte_count="$(du -sb "$stage" | awk '{print $1}')"
  manifest_sha256="$(sha256sum "$local_manifest" | awk '{print $1}')"
  archived_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cp "$local_manifest" "$stage/.archive_manifest.sha256"
  printf '{"schema_version":"quant_lab_nas_redacted_archive_receipt.v1","day":"%s","source":"%s:%s","file_count":%s,"byte_count":%s,"manifest_sha256":"%s","archived_at":"%s"}\n' \
    "$day" "$CLOUD_HOST" "$SOURCE_ROOT" "$file_count" "$byte_count" \
    "$manifest_sha256" "$archived_at" >"$stage/.archive_receipt.json"

  rm -rf -- "$backup"
  if [[ -d "$final" ]]; then
    mv "$final" "$backup"
  fi
  mv "$stage" "$final"
  rm -rf -- "$backup"
  printf '{"event":"archived","day":"%s","file_count":%s,"byte_count":%s,"manifest_sha256":"%s","archived_at":"%s"}\n' \
    "$day" "$file_count" "$byte_count" "$manifest_sha256" "$archived_at" \
    >>"$AUDIT_LOG"
  rm -f "$remote_manifest" "$local_manifest"
  trap - EXIT
done

cutoff="$(python3 - "$RETENTION_DAYS" <<'PY'
from datetime import UTC, datetime, timedelta
import sys

print((datetime.now(UTC) - timedelta(days=int(sys.argv[1]))).date().isoformat())
PY
)"
for archived in "$DEST_ROOT"/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]; do
  [[ -d "$archived" ]] || continue
  day="${archived##*/}"
  [[ "$day" < "$cutoff" ]] || continue
  removed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  removed_bytes="$(du -sb "$archived" | awk '{print $1}')"
  rm -rf -- "$archived"
  printf '{"event":"retention_removed","day":"%s","byte_count":%s,"removed_at":"%s"}\n' \
    "$day" "$removed_bytes" "$removed_at" >>"$AUDIT_LOG"
done
