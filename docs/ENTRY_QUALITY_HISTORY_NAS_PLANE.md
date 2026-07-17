# Entry Quality History NAS Research Compute Plane

## Scope and invariants

This plane migrates only `qlab build-entry-quality-history` historical compute to
the NAS. Current-time `build-entry-quality`, strategy opportunity current truth,
Paper ACK/Tracker, Promotion, Risk Permission, V5, exchange collection, Alpha
Factory, Factor Factory, cost calibration, and lake compaction remain unchanged.

Every task and result is constrained to:

```text
research_only=true
requires_cloud_validation=true
live_order_effect=none
```

There is no NAS-to-Lake mount, NAS Gold writer, exchange credential, automatic
promotion, live guard, risk input, or automatic cloud fallback. The cloud remains
the only publisher. Production remains `shadow` and `ABORT/ACTIVE_ABORT`.

## Mandatory pre-audit

### 1. Existing call chain

The legacy path was:

```text
quant-lab-entry-quality-history.timer (01:35 and 13:35 UTC)
  -> quant-lab-entry-quality-history.service
  -> qlab build-entry-quality-history
  -> cli.build_entry_quality_history_command
  -> run_with_job_metrics
  -> build_and_publish_entry_quality_history
  -> six full Lake reads
  -> historical compute
  -> eleven independent Gold publishes
  -> reports/ writes
```

The migrated path is:

```text
request timer
  -> seal_entry_quality_history_snapshot
  -> create_entry_quality_history_task
  -> signed pending task
NAS worker claim/lease/heartbeat
  -> signature verification
  -> content-addressed Blob sync
  -> lazy projected scan and streaming collect
  -> compute_entry_quality_history
  -> signed result bundle
cloud importer
  -> strict result verification
  -> publish_entry_quality_history_result
  -> one rollback-capable generation for eleven Gold tables and ten reports
  -> completed/imported state
```

### 2. Inputs

Only these six datasets are sealed:

| Dataset | Purpose |
| --- | --- |
| `silver/v5_trade_event` | actual entry and realized outcome evidence |
| `silver/v5_order_lifecycle` | fallback actual entry/fill evidence |
| `silver/market_bar` | pre-entry range and forward outcome bars |
| `silver/v5_candidate_event` | candidate decisions and factor context |
| `gold/v5_candidate_label` | candidate forward labels |
| `gold/cost_bucket_daily` | research roundtrip cost evidence |

No current advisory, permission, Paper, proposal, ACK, Tracker, promotion, API
auth, or exchange-private dataset is included.

### 3. Input time ranges

All boundaries are UTC and use half-open intervals.

| Input | Sealed range |
| --- | --- |
| trade, lifecycle, candidate | `[start_date 00:00, end_date + 1 day)` |
| market bars | `[start_date - 24h, end_date + 1 day + max(PULLBACK_HORIZON_HOURS))` |
| candidate labels | `[start_date, end_date + 1 day + max horizon)` |
| costs | `[start_date, end_date + 1 day)` |

The maximum forward horizon is imported from `PULLBACK_HORIZON_HOURS`; it is not
duplicated in the Research Plane. The file index is refreshed for all six inputs
on every request. If indexed bounds are absent, only the dataset's time column is
scanned to derive file bounds; the cloud does not materialize a full DataFrame.

### 4-6. Outputs, schemas, and keys

All outputs include common provenance:

```text
contract_version, schema_version, quant_lab_git_commit, source_version,
generated_at_utc, generated_from_bundle_id, as_of_date, window_hours,
source, mode, start_date, end_date, window_mode, cost_mode
```

All use `window_replace`, window keys `(window_mode, cost_mode)`, and
`empty_result_semantics=clear_window`.

| Gold dataset | Dataset-specific schema | Primary key after common window identity |
| --- | --- | --- |
| `v5_entry_quality_history_missed_low_audit` | event/run, symbol, entry timestamp/price/reason, pre-low deltas, 24h range position, realized net, exit, diagnosis | `source_event_key,symbol,entry_ts` |
| `v5_entry_quality_history_missed_low_by_symbol` | group, sample/loss/profit counts, diagnosis counts, average location/net, diagnosis mix | `group_key` |
| `v5_entry_quality_history_missed_low_by_entry_reason` | same aggregate schema as by-symbol | `group_key` |
| `v5_entry_quality_history_late_entry_chase_shadow` | strategy/source/event/candidate, symbol/time/price, 12h/24h location, F4/F5, risk/block shadow flags, realized/24h/48h outcome | `source_type,source_event_key` |
| `v5_entry_quality_history_late_entry_chase_threshold_sensitivity` | threshold, block/loss/profit counts, false-positive rate, blocked/unblocked averages, always-false live-ready flag, advisory | `threshold_bps` |
| `v5_entry_quality_history_pullback_reversal_shadow` | rule/candidate/event, market regime/risk, pullback conditions, F4/F5, selected cost and quality, horizon, gross/net/MFE/MAE/win/label status | `source_event_key,horizon_hours` |
| `v5_entry_quality_history_pullback_by_symbol` | group identity, candidate/symbol/regime/horizon, sample/complete counts, mean/median/P25/win/MFE/MAE, cost mix, research decision/reasons | `group_type,group_key` |
| `v5_entry_quality_history_pullback_by_regime` | same aggregate schema | `group_type,group_key` |
| `v5_entry_quality_history_pullback_by_horizon` | same aggregate schema | `group_type,group_key` |
| `v5_entry_quality_history_anti_leakage_check` | check name, PASS/FAIL, violation count, detail | `check_name` |
| `v5_entry_quality_history_metrics` | row counts, anti-leakage status, `ready_for_live_rows=0`, metrics JSON | window identity only |

The full ordered columns and Polars dtypes are authoritative in
`ENTRY_QUALITY_HISTORY_OUTPUT_SPECS`. The Result Manifest carries and the importer
recomputes an ordered schema fingerprint for every output.

### 7. Legacy publish and empty results

Legacy `_publish_history()` replaced an entire dataset with a schema-carrying
empty frame, otherwise upserted by a table-specific key. That made empty output
clear stale rows, but eleven writes could be observed independently.

The cloud publisher now first removes only rows matching the requested
`window_mode + cost_mode`, appends the replacement, deduplicates by the full key,
stages all eleven datasets and ten reports, then switches them as one
rollback-capable generation. Empty output therefore clears that window while
preserving other modes/cost modes. Every table contains identical
`generation_id`, `snapshot_id`, and `task_id` metadata before the generation
pointer is switched.

### 8. Reports

The historical compute renders exactly:

```text
missed_low_audit.csv
missed_low_by_symbol.csv
missed_low_by_entry_reason.csv
late_entry_chase_threshold_sensitivity.csv
pullback_reversal_by_symbol.csv
pullback_reversal_by_regime.csv
pullback_reversal_by_horizon.csv
anti_leakage_check.csv
entry_quality_historical_metrics.json
entry_quality_historical_summary.md
```

They are produced on NAS, signed in the result manifest, validated on cloud, and
switched with the Gold generation. No raw snapshot input is returned.

### 9. Window modes

* `full`: use the explicit start/end dates unchanged.
* `recent_7d`: clamp start to no earlier than `end_date - 6 days`.
* `recent_30d`: clamp start to no earlier than `end_date - 29 days`.
* `walk_forward`: retain the explicit range and apply point-in-time ordering as
  an anti-leakage gate; it does not permit future labels at decision time.

### 10. Cost modes

Both modes preserve existing numerical behavior: choose the maximum available
symbol roundtrip estimate from `roundtrip_all_in_cost_bps`, or twice a one-way
P75/selected estimate, with a 30 bps floor. Missing cost is marked degraded.

`conservative` is the scheduled production research namespace. `quant_lab` is a
separate historical window namespace for explicit comparison. It currently uses
the same conservative resolver; this migration intentionally does not invent a
weaker cost path or change research numbers.

### 11. Anti-leakage gates

All ten checks must exist, all must be `PASS`, and total violations must be zero:

1. `history_window_respected`
2. `label_ts_after_decision_ts`
3. `forward_label_end_boundary`
4. `candidate_label_identity`
5. `market_future_data_excluded`
6. `closed_bar_inputs_only`
7. `walk_forward_semantics`
8. `horizon_completion`
9. `bundle_source_identity`
10. `read_only_no_live_action`

`WARN`, `BLOCK`, `FAIL`, missing checks, missing rows, or a non-zero violation
reject the complete result. A complete forward outcome now requires a bar at or
after the full target horizon; merely having any future bar is insufficient.

### 12-13. Pure compute and side effects

`compute_entry_quality_history()` accepts six DataFrames and returns
`EntryQualityHistoryArtifacts`; it reads no Lake path and writes no file.

Side effects are isolated to:

* cloud Snapshot sealing and queue/status writes;
* NAS Blob cache, local snapshot, work, result, and status writes;
* cloud importer validation events, generation publish, report switch, and queue
  archive moves;
* explicit legacy local fallback, only when its environment gate is enabled.

### 14. Provenance

The signed Snapshot and Task bind the full 40-character quant-lab commit,
selected V5 bundle ID, Entry Quality schema, task parameters, file hashes, and
Snapshot digest. Derived rows retain the existing short commit prefix and bundle
ID; cloud validation checks that the prefix belongs to the full task commit.
Result and receipt bind both cloud and Worker full commits. A Worker whose build
commit differs from the task fails with `worker_code_mismatch`.

### 15. Compatibility tests

The original behavior tests remain unchanged, including threshold reports,
recent-7d filtering, pullback/anti-leakage, and empty-window stale cleanup. New
fixtures compare the pure function against the legacy implementation for all four
window modes and both cost modes, including eleven tables, nulls, values, schemas,
warnings, CSVs, metrics JSON, summary, and anti-leakage.

### 16. Readers

Web readers consume the threshold sensitivity, three pullback aggregates,
anti-leakage, and metrics Gold datasets. Daily Expert Export includes those same
historical research surfaces under its existing row caps. AI Research consumes
them only through the already sealed Expert Pack evidence; it does not read NAS
work directories. Web V2 additionally reads only the cloud queue status JSON to
show compute progress. Current Truth readers remain cloud-local.

## Contracts and validation

Research contracts are independent of Export Pack contracts and use Pydantic
`extra="forbid"`. Task and Snapshot use the cloud task key. Manifest and receipt
use the NAS Worker key. Canonical JSON excludes only the signature field.

The cloud importer verifies, before any publish: both signatures/key IDs; task,
snapshot, digest, commit, schema, parameter and bundle identity; exact 11-output
and 10-report sets; total size cap; safe non-symlink paths; each SHA and size;
actual Parquet metadata row count; exact ordered columns/dtypes and recomputed
schema fingerprint; output window/provenance; complete Anti-Leakage PASS; no
`LIVE_SMALL_READY`; cache byte accounting; receipt totals; and supersession.

## Queue and recovery

The queue is independent at `/var/lib/quant-lab/research_queue`. Claim is an
atomic pending-to-running rename and writes a claim epoch. Heartbeats renew the
lease without regressing a newer task state. An expired lease is recovered only
when the status SHA is still unchanged, no final result exists, and max attempts
remain. An uploaded result is left for cloud validation rather than recomputed.

If Gold is committed but queue finalization is interrupted, status remains
`publishing/finalize_pending`; the next importer run verifies the generation and
finishes archive/status moves. It is never mislabeled rejected. Duplicate signed
results are idempotent; conflicting duplicate payloads are rejected.

## systemd

Install `deploy/systemd/research-plane.env.example` as
`/etc/quant-lab/research-plane.env`, replace paths/key IDs, mode `0640`, owned by
root and the service group.

```bash
sudo cp deploy/systemd/quant-lab-entry-quality-history-request.* /etc/systemd/system/
sudo cp deploy/systemd/quant-lab-research-result-import.* /etc/systemd/system/
sudo cp deploy/systemd/quant-lab-entry-quality-history.service /etc/systemd/system/
sudo systemctl daemon-reload
```

The Request timer stays at 01:35/13:35 UTC with CPU 30%, MemoryHigh 500 MB,
MemoryMax 900 MB and one Polars thread. The Import timer runs every minute with
CPU 40%, MemoryHigh 700 MB, MemoryMax 1200 MB and one Polars thread.
The Request unit sees the Lake read-only except for
`ops/lake_file_index`, whose metadata membership is refreshed before sealing;
Silver and Gold input directories remain read-only.

The legacy service has an `ExecCondition` on
`QUANT_LAB_LOCAL_ENTRY_QUALITY_HISTORY_ENABLED=1`; its timer remains disabled. A
NAS outage leaves pending work and never starts local historical compute.

## NAS worker deployment

Use `deploy/nas_research_worker`. The dedicated container runs as UID/GID 10004,
with 3 CPUs, 256 PIDs, a read-only root, no-new-privileges, 512 MB noexec tmpfs,
rotated logs, and an 8 GB default memory limit. Acceptance still requires observed
peak RSS below 6 GB. It shares `/volume1/docker/quant-runtime/heavy-job.lock` with
the Export Worker. The dedicated `10004:10004` user receives supplemental group
`10002` only for that setgid runtime directory; the shared lock is kept `0660`.

Create a dedicated cloud SSH account with no sudo and queue-only group access.
Use an outbound-only NAS key and an `authorized_keys` `restrict` option. Pin the
qyun2 host key. Keep SSH, task verification, and Worker signing keys under Compose
read-only secrets, never in `.env`, Git, image layers, logs, or result bundles.

```bash
cd deploy/nas_research_worker
cp .env.example .env
# Set the exact full deployed SHA in BUILD_GIT_COMMIT.
docker compose build --pull
docker compose run --rm -e RUN_ONCE=true quant-research-worker
```

First validate the returned result without publishing:

```bash
COMMIT="$(git -C /opt/quant-lab rev-parse HEAD)"
sudo -u quantlab env QUANT_LAB_NAS_RESEARCH_ENABLED=1 \
  /opt/quant-lab/.venv/bin/qlab \
  import-entry-quality-history-results \
  --lake-root /var/lib/quant-lab/lake \
  --queue-root /var/lib/quant-lab/research_queue \
  --task-public-key /etc/quant-lab/research-task-public.pem \
  --task-key-id cloud-research-v1 \
  --worker-public-key /etc/quant-lab/nas-research-worker-public.pem \
  --worker-key-id nas-research-v1 \
  --quant-lab-commit "$COMMIT" \
  --validate-only
```

This path does not write status, move queue entries, or publish Gold. After it
passes, run one importer service, compare all eleven tables/reports with the
legacy fixture, run two full shadow cycles, and only then enable the new Request
and Import timers.

## Web V2

Operations displays task/snapshot/window/mode/cost, state, Worker, heartbeat,
input/download/cache, output rows, Anti-Leakage, cloud import, generation, and
error. Missing queue means `idle`; unreadable state means `not_observable`; NAS
offline means waiting. The UI never reports success or invokes local fallback.

## Manual rollback

Rollback is always an operator action:

```text
1. Stop quant-research-worker.
2. Disable/stop the new Request timer.
3. Disable/stop the Result Import timer.
4. Set QUANT_LAB_NAS_RESEARCH_ENABLED=0.
5. During an approved maintenance window only, set
   QUANT_LAB_LOCAL_ENTRY_QUALITY_HISTORY_ENABLED=1.
6. Enable/run the legacy timer or service.
```

Do not automatically move this workload back to qyun2. A rollback does not alter
V5, Risk Permission, current Entry Quality, Paper state, or exchange positions.
