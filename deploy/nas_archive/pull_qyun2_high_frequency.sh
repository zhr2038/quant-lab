#!/usr/bin/env bash
set -euo pipefail

CLOUD_HOST="${QUANT_ARCHIVE_CLOUD_HOST:-qyun2.hrhome.top}"
CLOUD_PORT="${QUANT_ARCHIVE_CLOUD_PORT:-22}"
CLOUD_USER="${QUANT_ARCHIVE_CLOUD_USER:-quant-research}"
SSH_KEY="${QUANT_ARCHIVE_SSH_KEY:-/volume1/docker/quant-research/secrets/id_ed25519}"
KNOWN_HOSTS="${QUANT_ARCHIVE_KNOWN_HOSTS:-/volume1/docker/quant-research/secrets/known_hosts}"
SOURCE_ROOT="${QUANT_ARCHIVE_SOURCE_ROOT:-/var/lib/quant-lab/lake/archive/high_frequency}"
DEST_ROOT="${QUANT_ARCHIVE_DEST_ROOT:-/volume1/docker/quant-archive/qyun2/high-frequency}"
AUDIT_ROOT="${QUANT_ARCHIVE_AUDIT_ROOT:-/volume1/docker/quant-archive/qyun2/audit}"
REMOTE_PRUNE_SCRIPT="${QUANT_ARCHIVE_REMOTE_PRUNE_SCRIPT:-/opt/quant-lab/deploy/nas_archive/prune_verified_high_frequency_archive.py}"
TRANSFER_TIMEOUT_SECONDS="${QUANT_ARCHIVE_TRANSFER_TIMEOUT_SECONDS:-10800}"
DATASETS=(
  "bronze/okx_public_ws"
  "silver/orderbook_snapshot"
  "silver/trade_print"
)

case "$DEST_ROOT" in
  /volume1/docker/quant-archive/qyun2/*) ;;
  *) echo "unsafe destination root: $DEST_ROOT" >&2; exit 2 ;;
esac
case "$TRANSFER_TIMEOUT_SECONDS" in
  ''|*[!0-9]*) echo "invalid transfer timeout" >&2; exit 2 ;;
esac

mkdir -p "$DEST_ROOT" "$AUDIT_ROOT"
exec 9>"$AUDIT_ROOT/high-frequency.lock"
flock -n 9 || exit 0

SSH=(
  ssh -i "$SSH_KEY" -p "$CLOUD_PORT" -o BatchMode=yes
  -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3
  -o "UserKnownHostsFile=$KNOWN_HOSTS"
)
RSYNC_SSH="ssh -i $SSH_KEY -p $CLOUD_PORT -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o UserKnownHostsFile=$KNOWN_HOSTS"
REMOTE="$CLOUD_USER@$CLOUD_HOST"
AUDIT_LOG="$AUDIT_ROOT/high-frequency.jsonl"
today_utc="$("${SSH[@]}" "$REMOTE" date -u +%F)"

for dataset in "${DATASETS[@]}"; do
  source_dataset="$SOURCE_ROOT/$dataset"
  mapfile -t source_days < <(
    "${SSH[@]}" "$REMOTE" \
      "find '$source_dataset' -mindepth 1 -maxdepth 1 -type d -name 'date=*' -printf '%f\\n' 2>/dev/null" \
      | LC_ALL=C sort
  )

  for date_dir in "${source_days[@]}"; do
    [[ "$date_dir" =~ ^date=([0-9]{4}-[0-9]{2}-[0-9]{2})$ ]] || continue
    day="${BASH_REMATCH[1]}"
    [[ "$day" < "$today_utc" ]] || continue

    source="$source_dataset/$date_dir"
    remote_manifest="$(mktemp "$AUDIT_ROOT/.hf-remote-manifest.XXXXXX")"
    local_manifest="$(mktemp "$AUDIT_ROOT/.hf-local-manifest.XXXXXX")"
    trap 'rm -f "$remote_manifest" "$local_manifest"' EXIT

    "${SSH[@]}" "$REMOTE" \
      "cd '$source' && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum" \
      >"$remote_manifest"
    [[ -s "$remote_manifest" ]] || {
      rm -f "$remote_manifest" "$local_manifest"
      trap - EXIT
      continue
    }

    manifest_sha256="$(sha256sum "$remote_manifest" | awk '{print $1}')"
    batch_root="$DEST_ROOT/$dataset/$date_dir"
    final="$batch_root/batch=$manifest_sha256"
    stage="$batch_root/.batch=$manifest_sha256.partial"
    mkdir -p "$batch_root"

    if [[ -L "$final" ]]; then
      echo "unsafe archive target symlink: $final" >&2
      exit 1
    fi
    if [[ -e "$final" && \
          ( ! -f "$final/.archive_receipt.json" || \
            ! -f "$final/.archive_manifest.sha256" ) ]]; then
      echo "incomplete existing archive target: $final" >&2
      exit 1
    fi

    if [[ ! -d "$final" ]]; then
      case "$stage" in
        "$DEST_ROOT"/*/.batch=*.partial) ;;
        *) echo "unsafe stage path: $stage" >&2; exit 2 ;;
      esac
      rm -rf -- "$stage"
      mkdir -p "$stage"
      timeout "$TRANSFER_TIMEOUT_SECONDS" rsync \
        --archive --partial --delay-updates --safe-links \
        -e "$RSYNC_SSH" "$REMOTE:$source/" "$stage/"
      (
        cd "$stage"
        find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
      ) >"$local_manifest"
      cmp --silent "$remote_manifest" "$local_manifest" || {
        echo "high-frequency archive checksum mismatch: $dataset $day" >&2
        exit 1
      }

      file_count="$(wc -l <"$local_manifest" | tr -d ' ')"
      byte_count="$(du -sb "$stage" | awk '{print $1}')"
      archived_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      cp "$local_manifest" "$stage/.archive_manifest.sha256"
      printf '{"schema_version":"quant_lab_nas_high_frequency_archive_receipt.v1","dataset":"%s","day":"%s","source":"%s:%s","file_count":%s,"byte_count":%s,"manifest_sha256":"%s","archived_at":"%s"}\n' \
        "$dataset" "$day" "$CLOUD_HOST" "$source" "$file_count" "$byte_count" \
        "$manifest_sha256" "$archived_at" >"$stage/.archive_receipt.json"
      mv "$stage" "$final"
      printf '{"event":"archived","dataset":"%s","day":"%s","file_count":%s,"byte_count":%s,"manifest_sha256":"%s","archived_at":"%s"}\n' \
        "$dataset" "$day" "$file_count" "$byte_count" "$manifest_sha256" \
        "$archived_at" >>"$AUDIT_LOG"
    else
      (
        cd "$final"
        find . -type f ! -name '.archive_manifest.sha256' ! -name '.archive_receipt.json' \
          -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
      ) >"$local_manifest"
      cmp --silent "$remote_manifest" "$local_manifest" || {
        echo "existing high-frequency archive checksum mismatch: $dataset $day" >&2
        exit 1
      }
    fi

    "${SSH[@]}" "$REMOTE" sudo -u quantlab -n /usr/bin/python3 \
      "$REMOTE_PRUNE_SCRIPT" "$dataset" "$day" "$manifest_sha256" --apply
    pruned_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"event":"source_pruned_after_verified_archive","dataset":"%s","day":"%s","manifest_sha256":"%s","pruned_at":"%s"}\n' \
      "$dataset" "$day" "$manifest_sha256" "$pruned_at" >>"$AUDIT_LOG"

    rm -f "$remote_manifest" "$local_manifest"
    trap - EXIT
  done
done
