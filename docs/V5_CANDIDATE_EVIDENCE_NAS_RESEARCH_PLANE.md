# V5 Candidate Evidence NAS Research Compute Plane

This runbook covers the research-only migration of incremental V5 Candidate
Labels and Strategy Evidence Samples to the signed NAS Research Plane. The
implementation baseline is `222386a4e1883ce80d32bf288a3b617ce4a7a0a9`.

The NAS computes only two deltas:

```text
v5_candidate_label_delta
strategy_evidence_sample_delta
```

qyun2 remains the sole authority for Candidate Quality, Candidate Outcome
Summary, Strategy Evidence aggregation and Decision Ladder, all PAPER/LIVE
readiness decisions, six-table Gold publication, the generation pointer, and
downstream control jobs. The task is diagnostic/research-only, has zero live
notional, and has no exchange credential or order effect.

## Legacy and new call chains

The legacy hourly path was:

```text
build-v5-candidate-labels --mode incremental --lookback-days 8
  -> read Candidate Event, closed Market Bar and Run Summary
  -> compute seven label horizons
  -> derive Quality and Outcome Summary
  -> write three Candidate Gold tables

build-strategy-evidence --mode incremental --lookback-days 8
                        --skip-historical-outcomes
  -> read Candidate Event and Candidate Label
  -> compute Evidence Samples
  -> aggregate Evidence Summary and Quality
  -> apply the cloud Decision Ladder
  -> write three Strategy Evidence Gold tables
```

The migrated path is:

```text
qyun2 request service
  -> projection-scoped fingerprint
  -> fast verification of the current six-table generation
  -> signed immutable snapshot and task
NAS worker
  -> computing_labels
  -> computing_samples
  -> strict 30-check anti-leakage report
  -> signed sharded result and receipt
qyun2 importer
  -> capacity gates before global scans
  -> task/snapshot/result/receipt and previous-generation validation
  -> cloud-only Quality, Outcome, Summary, Quality and Decision derivation
  -> atomic six-table publication and generation pointer
```

The first closed 1H bar strictly after the event remains the Decision Bar. The
exact horizons remain `4, 8, 12, 24, 48, 72, 120` hours. The target is the first
bar at or after Decision Time plus the horizon. MFE/MAE use the inclusive path
from Decision Bar through Label Bar. Insufficient future data remains a
`partial/future_bar_unavailable` row. Cost remains sourced from the signed
Candidate Event, including its signed raw payload fallback.

## Keys and empty-result behavior

| Dataset | Managed primary key |
| --- | --- |
| V5 Candidate Label | `strategy, candidate_id, horizon_hours` |
| V5 Candidate Quality Daily | `strategy, date` |
| V5 Candidate Outcome Summary | `strategy, date, block_reason, strategy_candidate, symbol, horizon_hours` |
| Strategy Evidence Sample | `source, strategy, source_type, candidate_id, symbol, strategy_candidate, horizon_hours, source_event_key` |
| Strategy Evidence Summary | `source, strategy, evidence_version, as_of_date, strategy_candidate, symbol, regime_state, horizon_hours` |
| Strategy Evidence Quality | `source, strategy, evidence_version, as_of_date, severity, warning_type` |

Candidate Quality retains the legacy one-row warning behavior for an empty
scope. Empty Outcome Summary and Strategy Evidence Summary inputs preserve the
previous accepted managed scope. Shared Strategy Evidence datasets retain
rows owned by other `source` values and other dates.

## Identity and bounded inputs

The fingerprint includes only the signed projection that can affect this task:

- Candidate Events in the eight-day event window;
- closed Market Bars for Candidate symbols, using the selected timeframe and
  extending far enough for the 120-hour horizon;
- all Run Summary rows in the same event window, including runs that have no
  Candidate Event;
- exact parameters, schema versions, projection version, and full commit SHA.

Candidate Event run IDs and sealed Run Summary run IDs are recorded
independently in the v2 manifest. Any Run Summary change inside the event
window changes identity even when that run has no Candidate Event; only
out-of-window Run Summary rows remain projection-irrelevant. Unrelated event
dates, market symbols, timeframes, and unclosed bars do not change identity.
The snapshot stores projected file hashes, schemas, row counts, timestamp
bounds, compressed and Parquet-uncompressed sizes. Rehydrate reproduces the
same file identity and preserves the original manifest bytes, signature, and
seal; a source change rejects rehydrate. New work uses
`v5_candidate_evidence_projection.v2` and
`quant_lab_v5_candidate_evidence_snapshot.v2`; v1 snapshots remain auditable
but cannot be executed silently with v2 worker semantics.

Default independent limits are:

```text
snapshot compressed             512 MiB
snapshot estimated uncompressed   1 GiB
input rows                         5,000,000
result compressed                256 MiB
result uncompressed              512 MiB
partition compressed              64 MiB
partition uncompressed           128 MiB
result files                       5,000
```

The worker checks snapshot compressed/uncompressed size, rows, and free disk
before compute. The importer checks file count, total compressed size, per-file
compressed and uncompressed size, and total uncompressed size before Unique,
GroupBy, or Join scans.

## Queue and publication rules

An identical pending or running fingerprint is coalesced. A newer fingerprint
cancels an older pending task but never kills a running task. At most one
pending successor is retained. A late result is rejected when a newer task in
the same scope exists, and publication also requires an exact previous
generation ID and digest.

The importer stages all six datasets, validates their full keys and safety
scope, writes generation sidecars, then commits all directories and the pointer
through one recoverable journal. A failed post-commit verification rolls every
dataset and the pointer back. Re-import of the same accepted result is
idempotent.

## Scheduling and gates

`quant-lab-v5-candidate-evidence-request.timer` runs hourly at minute 20 UTC
with up to two minutes of randomized delay. This minute was selected after
auditing Telemetry Sync, 15-minute Market Backfill and Feature Publish jobs, the
one-minute Research Importer, and the Factor Factory request at minute 38.

The request service is limited to one Polars thread, 30% CPU, 500 MB
`MemoryHigh`, and 900 MB `MemoryMax`. It only fingerprints, fast-verifies,
materializes/rehydrates a snapshot, and enqueues a task.

Enable the plane deliberately:

```text
qyun2:
  QUANT_LAB_NAS_RESEARCH_ENABLED=1
  QUANT_LAB_NAS_V5_CANDIDATE_EVIDENCE_ENABLED=1

NAS worker:
  QUANT_RESEARCH_V5_CANDIDATE_EVIDENCE_ENABLED=1

normal production local fallback:
  QUANT_LAB_LOCAL_V5_CANDIDATE_EVIDENCE_ENABLED=0
```

During pre-cutover shadow only, the legacy refresh service explicitly enables
the local command gate so the old accepted path remains available. After two
successful shadows and downstream checks, remove the two legacy commands and
that temporary environment line before enabling the new request timer. Never
run both writers concurrently.

## Acceptance

1. Deploy one exact commit to qyun2 and the NAS image.
2. Run the NAS worker once with `RUN_ONCE=true` while the request timer remains
   disabled.
3. Run the importer with `--validate-only` and confirm 30 PASS/0 checks.
4. Compare all six normalized Gold tables with the legacy fixed fixture and a
   production shadow; ignore only generation/host/audit identity and
   `created_at`.
5. Perform one formal import, then two additional shadow cycles.
6. Verify Alpha Discovery, Trade-Level Judgment, and Paper Pipeline against the
   accepted generation.
7. Remove the two old refresh commands, redeploy the same final main SHA, and
   enable the new request timer.
8. Verify a no-change request and a released-snapshot rehydrate cycle.

Record cold/warm cache bytes, runtime, CPU time, peak RSS, temporary disk,
result compressed/uncompressed bytes, importer validation/publication time,
DuckDB spill, and all input/output row counts in the final acceptance report.

## Manual rollback

1. Disable and stop `quant-lab-v5-candidate-evidence-request.timer`.
2. Stop creating new Candidate Evidence tasks; let a claimed task finish or
   cancel it explicitly.
3. Keep the shared importer enabled for other task types.
4. Set the NAS Candidate Evidence worker gate to `0` and keep automatic local
   fallback at `0`.
5. Restore the two legacy commands to V5 Research Refresh.
6. During an approved maintenance window only, set
   `QUANT_LAB_LOCAL_V5_CANDIDATE_EVIDENCE_ENABLED=1` and run the legacy path.

NAS unavailability always leaves the last accepted cloud generation in place;
it never triggers cloud recomputation automatically.
