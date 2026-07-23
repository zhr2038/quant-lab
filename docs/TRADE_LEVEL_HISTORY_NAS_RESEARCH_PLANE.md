# Trade-Level Historical Similarity NAS Research Plane

This runbook covers the research-only migration of Trade Opportunity Labels
and causal Trade-Level Similarity Outcomes to the signed NAS Research Plane.
The migration does not move Trade-Level Judgment or any trading control.

The NAS may return only:

```text
trade_opportunity_label
trade_level_similarity_outcome
```

qyun2 remains the sole authority for Trade Opportunity Event derivation,
point-in-time Risk Permission context, Judgment, False Block Audit, V5 Trade
Learning Sample, Attribution, Opportunity Cost, Bucket Policy, Opportunity
Queue, all review/allow decisions, order limits, and Gold Current Truth. The
task is diagnostic-only and research-only, has zero live notional, and has no
exchange credential or order effect.

## Legacy and migrated call chains

The legacy path derives and writes all history and control tables in one cloud
process:

```text
qlab build-trade-level-judgment
  -> derive Trade Opportunity Event
  -> build 4h/8h/24h labels
  -> build historical similarity
  -> run the two-stage cloud judgment and bucket-policy loop
  -> write thirteen Gold datasets
```

The migrated path is:

```text
qyun2 request service
  -> Fast Verify current Candidate Evidence Generation
  -> derive authoritative Event with point-in-time Risk Permission
  -> compute the full-history input fingerprint
  -> no-change Fast Verify
  -> seal and sign a two-dataset immutable snapshot
NAS worker
  -> verify task, snapshot, commit, and image/repository provenance
  -> compute labels by Symbol
  -> compute causally available similarity by Symbol
  -> run 32 anti-leakage checks
  -> sign a sharded result and receipt
qyun2 importer
  -> apply capacity gates before global scans
  -> verify task, snapshot, result, receipt, and previous generation
  -> causally recompute the result from the signed snapshot
  -> atomically replace Event, Label, and Similarity Gold
qyun2 control plane
  -> consume the last Fast-Verified accepted history generation
  -> run Judgment, Audit, Learning, Attribution, Cost, Policy, and Queue
```

The cloud control chain never waits synchronously for NAS compute. NAS
unavailability leaves the last accepted history generation in place and never
enables a local fallback.

## Point-in-time and causal rules

For an event at time `T`, a historical event is eligible only when its
`decision_ts < T`. Events sharing the same timestamp never reference one
another, and the current event never references itself.

The selected prior outcome is the largest available horizon in this order:

```text
24h -> 8h -> 4h
```

An outcome is available only when its `label_<horizon>h_available_at <= T`.
Candidate `label_ts` is authoritative when present; otherwise availability is
derived as `decision_ts + horizon` and explicitly records
`derived_from_horizon`. `created_at` is never treated as label availability.

Risk Permission is selected on qyun2, not NAS:

1. signed Permission context embedded in the Candidate Event;
2. the latest Permission with `as_of_ts <= decision_ts`;
3. `UNKNOWN/MISSING` when no historical context exists.

A future Permission is invalid. Missing Permission remains fail-closed and
cannot produce `MICRO_CANARY_ALLOW` or a non-zero order limit.

## Identity, snapshot, and no-change

The input fingerprint binds the exact commit, as-of day, PARITY_FULL mode,
schema and availability-policy versions, derived Event digest, Candidate Label
dataset hash, verified Candidate Evidence generation identity, row counts, and
timestamp bounds.

Snapshot identity contains only immutable compute inputs:

```text
derived Event digest
Candidate Label dataset hash
Candidate Evidence generation identity
schema versions
availability policy
commit
PARITY_FULL mode
```

It does not contain the previous history generation, request time, or worker
identity. Task identity additionally binds the snapshot, previous generation
ID/digest, signing key ID, and exact commit.

The snapshot contains exactly:

```text
cloud/trade_opportunity_event/data.parquet
gold/v5_candidate_label/data.parquet
```

Before returning `already_current`, the request path Fast Verifies the pointer,
generation digest, all three Gold datasets, sidecars, schemas, primary keys,
managed columns, row counts, dataset hashes, safety fields, and Candidate
Evidence generation binding. Any failure returns
`generation_integrity_failed`; it must not materialize a snapshot or enqueue a
task.

Released snapshots preserve their manifest, signature, seal, and release
marker. Rehydrate takes the snapshot lock, recreates the exact two files in a
partial directory, compares every file attribute with the retained manifest,
and installs the payload atomically. Any source, schema, hash, row-count, or
timestamp-bound difference rejects rehydrate.

## Capacity and publication

Default independent limits are:

```text
snapshot compressed               2 GiB
snapshot estimated uncompressed   4 GiB
input rows                        10,000,000
result compressed                 1 GiB
result uncompressed               2 GiB
partition compressed             128 MiB
partition uncompressed           256 MiB
result files                      20,000
```

The worker reads one Symbol at a time, stages labels to local Parquet, releases
the source frame, then computes similarity and writes result shards
incrementally. Target NAS peak RSS is below 6 GiB. The importer uses one DuckDB
thread, a 512 MiB memory limit, and a writable spill directory; target importer
peak RSS is below 3 GiB.

Publication stages the authoritative cloud Event plus the two validated NAS
outputs, validates exact schemas and `event_id` primary keys, writes generation
sidecars, and commits all three datasets and the pointer through one durable
journal. A failed post-commit Fast Verify rolls back the transaction. Re-import
of the same accepted generation is idempotent.

## Scheduling and enable gates

`quant-lab-trade-level-history-request.timer` polls hourly at minute 50 UTC
with up to two minutes of randomized delay. The minute follows the Candidate
Evidence request, telemetry sync, and one-minute importer, while leaving time
before the next V5 Research Refresh cycle.

The request service uses:

```text
POLARS_MAX_THREADS=1
CPUQuota=30%
MemoryHigh=600M
MemoryMax=1G
```

Enable deliberately:

```text
qyun2:
  QUANT_LAB_NAS_RESEARCH_ENABLED=1
  QUANT_LAB_NAS_TRADE_LEVEL_HISTORY_ENABLED=1

NAS:
  QUANT_RESEARCH_TRADE_LEVEL_HISTORY_ENABLED=1

normal production:
  QUANT_LAB_LOCAL_TRADE_LEVEL_HISTORY_ENABLED=0
```

The NAS image must be rebuilt from the final main SHA. `/app/BUILD_GIT_COMMIT`,
image-provided `BUILD_GIT_COMMIT`, image-provided
`QUANT_RESEARCH_WORKER_COMMIT`, and mounted repository HEAD must all match
before the worker polls or claims a task. Do not place a runtime worker-commit
override in `.env`.

## Shadow and formal cutover

Before formal publication, run:

```text
request
-> NAS RUN_ONCE
-> 32/32 PASS
-> cloud --validate-only
-> formal import
-> Generation Fast Verify
-> no-change request
-> released-snapshot rehydrate
```

Then run one changed-input and one stable-input immediate Shadow. Because the
migration fixes historical future-label leakage, continue the legacy cloud
control writer for at least seven daily read-only Shadow reports. Each report
compares similarity sample count, mean, median, P25, hit rate, max adverse,
recent 7d mean, downstream decisions, review/allow/risk-block counts, and order
limits.

During Shadow:

- the legacy path must not overwrite the three accepted history Gold tables;
- Risk Permission must remain byte-semantically unchanged;
- no new `MICRO_CANARY_ALLOW` may be published;
- no event or bucket order limit may increase;
- automatic promotion remains false.

Only after seven clean daily reports may V5 Research Refresh switch to
`qlab build-trade-level-control`, the request timer be treated as the formal
source, and the legacy history writer be disabled. Until then,
`build-trade-level-legacy-control-shadow` is the transitional refresh command.

## Manual rollback

1. Disable and stop `quant-lab-trade-level-history-request.timer`.
2. Set the NAS Trade-Level History worker gate to `0`.
3. Leave the shared importer enabled for other Research Plane task types.
4. Keep the last Fast-Verified three-table history generation unchanged.
5. Restore the prior V5 Research Refresh unit and reload systemd.
6. During an explicitly approved maintenance window only, set
   `QUANT_LAB_LOCAL_TRADE_LEVEL_HISTORY_ENABLED=1` and run the legacy command.
7. Do not alter Risk Permission, live-order services, or order limits as part
   of the research-plane rollback.
