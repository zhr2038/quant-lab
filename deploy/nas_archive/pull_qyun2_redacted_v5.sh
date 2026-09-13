#!/usr/bin/env bash
set -euo pipefail

CLOUD_HOST="${QUANT_ARCHIVE_CLOUD_HOST:-qyun2.hrhome.top}"
CLOUD_PORT="${QUANT_ARCHIVE_CLOUD_PORT:-22}"
CLOUD_USER="${QUANT_ARCHIVE_CLOUD_USER:-quant-research}"
SSH_KEY="${QUANT_ARCHIVE_SSH_KEY:-/volume2/quant-lab/ops/private/id_ed25519}"
KNOWN_HOSTS="${QUANT_ARCHIVE_KNOWN_HOSTS:-/volume2/quant-lab/ops/private/known_hosts}"
SOURCE_ROOT="${QUANT_ARCHIVE_SOURCE_ROOT:-/var/lib/quant-lab/archive/v5/bundles}"
DEST_ROOT="${QUANT_ARCHIVE_DEST_ROOT:-/volume2/quant-lab/archive/current/qyun2/redacted-v5}"
AUDIT_ROOT="${QUANT_ARCHIVE_AUDIT_ROOT:-/volume2/quant-lab/archive/current/qyun2/audit}"
REMOTE_PRUNE_SCRIPT="${QUANT_ARCHIVE_REMOTE_PRUNE_SCRIPT:-/opt/quant-lab/deploy/nas_archive/prune_verified_redacted_v5_archive.py}"
VERIFY_RECEIPT_SCRIPT="${QUANT_ARCHIVE_VERIFY_RECEIPT_SCRIPT:-/volume2/quant-lab/ops/bin/verify_redacted_v5_archive_receipt.py}"
SOURCE_KEEP_DAYS="${QUANT_ARCHIVE_SOURCE_KEEP_DAYS:-2}"
TRANSFER_TIMEOUT_SECONDS="${QUANT_ARCHIVE_TRANSFER_TIMEOUT_SECONDS:-10800}"

case "$SOURCE_ROOT" in
  /var/lib/quant-lab/archive/v5/bundles) ;;
  *) echo "unsafe source root: $SOURCE_ROOT" >&2; exit 2 ;;
esac
case "$DEST_ROOT" in
  /volume2/quant-lab/archive/current/qyun2/*) ;;
  *) echo "unsafe destination root: $DEST_ROOT" >&2; exit 2 ;;
esac
case "$SOURCE_KEEP_DAYS" in
  ''|*[!0-9]*) echo "invalid source keep days" >&2; exit 2 ;;
esac
(( SOURCE_KEEP_DAYS >= 1 )) || { echo "source keep days must be positive" >&2; exit 2; }


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
PRUNE_AUDIT_LOG="$AUDIT_ROOT/redacted-v5-source-prune.jsonl"

verified_receipt_fields() {
  python3 "$VERIFY_RECEIPT_SCRIPT" \
    --receipt "$1" --manifest "$2" --archive-root "$3" --day "$4" \
    --source-host "$CLOUD_HOST" --source-root "$SOURCE_ROOT" --line-output
}

today_utc="$("${SSH[@]}" "$REMOTE" date -u +%F)"
source_cutoff="$(python3 - "$today_utc" "$SOURCE_KEEP_DAYS" <<'PY'
from datetime import date, timedelta
import sys

print((date.fromisoformat(sys.argv[1]) - timedelta(days=int(sys.argv[2]) - 1)).isoformat())
PY
)"
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

  if [[ ! -f "$final/.archive_receipt.json" || ! -f "$final/.archive_manifest.sha256" ]]; then
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
  fi

  if [[ "$day" < "$source_cutoff" ]]; then
    mapfile -t receipt_fields < <(
      verified_receipt_fields \
        "$final/.archive_receipt.json" "$final/.archive_manifest.sha256" "$final" "$day"
    )
    manifest_sha256="${receipt_fields[0]}"
    file_count="${receipt_fields[1]}"
    printf -v remote_prune_command \
      'sudo -n -u quantlab /usr/bin/python3 %q --source-root %q --day %q --expected-manifest-sha256 %q --expected-file-count %q --apply' \
      "$REMOTE_PRUNE_SCRIPT" "$SOURCE_ROOT" "$day" "$manifest_sha256" "$file_count"
    prune_result="$("${SSH[@]}" "$REMOTE" "$remote_prune_command")"
    printf '%s\n' "$prune_result" >>"$PRUNE_AUDIT_LOG"
  fi
  rm -f "$remote_manifest" "$local_manifest"
  trap - EXIT
done

# Historical records are retained permanently; capacity review is explicit.
echo "ARCHIVE_COMPLETE_HISTORY_RETAINED"
