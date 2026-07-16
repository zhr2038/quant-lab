# Expert Pack NAS Local Plane

## Scope and safety

This change moves Expert Pack materialization, validation, accepted storage, download, and
AI evidence reading to the NAS. It does not modify V5, exchange access, strategy rules,
risk permission, Paper ACK/Tracker state, or live order behavior.

Production invariants remain:

- quant-lab mode is `shadow`;
- Permission remains `ABORT/ACTIVE_ABORT`;
- Canary and Enforce remain disabled;
- NAS containers hold no exchange credentials;
- `live_order_effect=none` for Export and AI Research;
- no automatic cloud fallback is allowed.

## Pre-change audit

The original `export_daily_pack()` path was inspected before the split.

1. It refreshed or observed V5 state before export.
2. It optionally published Risk Permission.
3. It loaded the report snapshot into DataFrames.
4. It published multiple Current/Gold diagnostic snapshots as an export side effect.
5. It built every CSV/JSON/Markdown/PNG member in one in-memory dictionary.
6. It embedded a redacted V5 bundle.
7. It scanned all members, assembled the ZIP, wrote cloud indexes, and retained ZIP files.
8. Streamlit and Web V2 inspected cloud ZIP files and exposed cloud download endpoints.
9. AI task construction scanned the cloud export directory and opened the latest ZIP.
10. The scheduled and Web-triggered services could each use 5-6 GB RSS.

Production measurements from the last cloud authoritative pack before the change:

| Metric | Old cloud path |
| --- | ---: |
| Export time | 442.101 s |
| Peak RSS | 6,395.609 MB |
| Snapshot load peak | 3,846 MB |
| Pack size | 88,952,711 bytes |
| Members | 275 |
| `publish_strategy_opportunity_advisory` | 160.737 s |
| `publish_trade_level_judgment` | 43.626 s |
| `build_members` | 211.321 s |

The export-only Current/Gold publishing steps remain cloud responsibilities and are no
longer executed by the NAS materializer. The NAS reads only a sealed immutable snapshot.

## Architecture

```text
qyun2 control plane
  Web/API -> lightweight request
  systemd snapshot worker -> signed immutable input snapshot
  export queue -> signed task
                       |
                       | NAS outbound SSH/SCP only
                       v
NAS quant-export-worker
  verify task + snapshot signatures
  content-addressed blob sync
  materialize reports to staging
  stream ZIP + validate locally
  atomic accepted directory + signed receipt
                       |
                       | small signed receipt only
                       v
qyun2 receipt importer -> verified cloud index/status

NAS quant-export-download -> accepted index -> Nginx X-Accel-Redirect/sendfile
NAS quant-ai-worker -> accepted index + receipt -> local ZIP evidence -> AI result only
```

The Expert Pack byte stream never crosses qyun2 after the snapshot inputs are pulled by
the NAS. qyun2 does not store, receive, inspect, or proxy the finished ZIP.

## Trust semantics

The following flags are deliberately separate:

- `authoritative_input_snapshot`: qyun2 sealed and signed the exact inputs.
- `nas_artifact_validated`: the NAS completed ZIP and evidence validation.
- `control_plane_receipt_verified`: qyun2 verified the NAS Ed25519 receipt and bindings.
- `download_ready`: all preceding conditions hold and `pack_state=accepted`.

`nas_artifact_validated` never means qyun2 recomputed the ZIP SHA. qyun2 validates the
signed receipt, task, snapshot, worker identity, commit, V5 SHA, and Acceptance Set.

## Export contracts

Strict Pydantic contracts live under `src/quant_lab/export_plane/`:

- `ExportSnapshotManifest`
- `ExportDatasetReference`
- `ExportTask`
- `ExportWorkerReceipt`
- `ExportPackManifest`
- `ExportValidationReport`
- `ExportPackIndexEntry`
- `ExportTaskStatus`

All reject extra fields. IDs, SHA256 values, full commit SHAs, UTC timestamps, and relative
paths are validated. Cloud tasks/snapshots and NAS receipts use separate Ed25519 keys.

## Cloud queue

Default root:

```text
/var/lib/quant-lab/export_queue/
  requests/{pending,processing,completed,failed,status}/
  {pending,running,completed,failed,expired,cancelled}/
  receipts/{inbox,imported,rejected}/
  snapshots/
  status/
  cloud_index.json
```

Web writes only a small request. `quant-lab-export-request.path` starts the constrained
snapshot worker. A task enters `pending` only after the snapshot is fully sealed.

## NAS storage

```text
/data/
  blobs/sha256/<prefix>/<sha256>
  snapshots/<snapshot_id>/
  work/<task_id>/
  accepted/YYYY/MM/DD/<pack_id>/
  rejected/
  status/worker.json
  audit/retention.jsonl
  accepted_index.json
```

Cloud snapshot inputs are copied into a group-readable, read-only tree rather than hard
linked, so later in-place source mutations cannot change a sealed Snapshot. After a signed
NAS receipt is verified, qyun2 deletes only the Snapshot byte tree and retains its manifest,
signature, task, receipt, and release marker for audit.

Blob downloads use `.partial`, size and SHA checks, and atomic rename. The snapshot tree
is assembled from verified blobs. Repeated snapshots reuse blobs and do not re-download
unchanged inputs.

Accepted packs are immutable and contain:

- the Expert Pack ZIP;
- `pack_manifest.json`;
- `validation_report.json`;
- `worker_report.json`;
- `snapshot_manifest.json`;
- `receipt.json`.

Rejected and partial packs never enter `accepted_index.json`.

The Download and AI containers mount the whole Export data root read-only. This is
intentional: `accepted_index.json` is atomically replaced, so a single-file bind mount
would remain attached to the old inode and silently serve stale Pack state.

## Local validation

The validator checks safe ZIP paths, no symlinks, no duplicate names, member count, member
size, total expanded size, compression ratio, required files, CRC, member SHA256, CSV row
counts, secret patterns, Snapshot ID, V5 SHA, Acceptance Set, full quant-lab commit, and
authoritative-input state. CRC, SHA, row count, and secret checks share one streaming pass.

## Download plane

The Python app only authenticates and maps Pack ID to the accepted index. It returns an
`X-Accel-Redirect`; unprivileged Nginx sends the file using `sendfile` and implements HTTP
Range. Python never opens or buffers the ZIP for download.

The service is LAN/VPN only and uses both Basic Authentication at Nginx and a short-lived
HMAC URL bound to Pack ID, Pack SHA, expiry, nonce, and key ID.

## AI compatibility

Cloud AI tasks now carry only:

- `source_pack_id`;
- `source_pack_sha256`;
- `source_snapshot_id`;
- `source_location=nas_accepted`.

The NAS AI Worker verifies the accepted index, ZIP SHA, signed receipt, Snapshot identity,
and download-ready trust state. It then builds the bounded evidence packet locally. Only
the structured AI result and a bounded evidence-member index return to qyun2; the Pack and
evidence bodies remain on NAS.

## Production switches

```text
QUANT_LAB_NAS_EXPORT_ENABLED=1
QUANT_LAB_LOCAL_EXPORT_ENABLED=0
QUANT_LAB_WEB_LOCAL_EXPORT_ENABLED=0
```

The old daily and Web export services remain for explicit emergency rollback only. Their
`ExecCondition` refuses to run unless the corresponding local-export switch is set to 1.
The old timer/path units must remain disabled in normal production.

## Deployment order

1. Deploy the same clean quant-lab commit to qyun2 and the NAS Worker image.
2. Generate separate cloud and NAS Ed25519 key pairs.
3. Install only the cloud private key on qyun2 and only its public key on NAS.
4. Install only the NAS private key on NAS and only its public key on qyun2/AI Worker.
5. Generate one 48-byte download secret and mount it read-only on qyun2 and NAS Download.
6. Create a low-privilege `quant-export` SSH account with queue-only filesystem access.
7. Install `export-plane.env`, the request path unit, and receipt-import timer on qyun2.
8. Build and start `quant-export-worker` on NAS.
9. Build and start `quant-export-download` on a LAN/VPN bind address only.
10. Rebuild `quant-ai-worker` with the read-only accepted Pack mount.
11. Disable the legacy daily timer and Web export path.
12. Submit one authoritative request and verify the complete signed round trip.

## Rollback

Rollback is explicit and manual:

1. Stop NAS Export and Download containers.
2. Set `QUANT_LAB_NAS_EXPORT_ENABLED=0`.
3. Set local export switches to 1 only during an approved maintenance window.
4. Re-enable the chosen legacy timer/path manually.
5. Restart API/Web.

There is no automatic fallback because an unnoticed cloud fallback would recreate the
memory and disk-pressure incident this change is designed to remove.

## Retention

Defaults are 90 days, 200 GB, and at least 30 packs. Packs not yet marked `ai_consumed`
are pinned. Deletions use the accepted index, never a broad directory scan, and append a
retention audit record. Web cannot delete packs.

## Verification commands

```bash
pytest -q
ruff check .
python -m compileall -q src deploy tools
git diff --check
npm --prefix frontend-bigscreen run build
docker compose -f deploy/nas_export_worker/docker-compose.yml config
docker compose -f deploy/nas_export_download/docker-compose.yml config
```

Live acceptance additionally checks cloud RSS, NAS peak RSS, cold/warm Blob downloads,
pack member parity, Range `206`, direct NAS traffic, signed receipt import, Web rendering,
AI local-pack hydration, and zero live-order side effects.
