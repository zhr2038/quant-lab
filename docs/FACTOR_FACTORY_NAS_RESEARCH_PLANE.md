# Factor Factory NAS Research Compute Plane

This runbook defines the research-only migration of `qlab build-factor-factory`
to the existing signed NAS Research Compute Plane. The cloud remains the only
authority for plans, snapshots, candidate derivation, Gold publication, and
generation state. The NAS has no Lake mount, exchange credential, live state,
promotion authority, or non-zero notional.

The implementation is deliberately limited to Factor Factory. Feature publish,
candidate labels, strategy evidence, paper tracking, risk permissions, V5, and
all order paths remain outside this change.

## Baseline audit

The implementation baseline was production commit
`6652c3669f09ad25ce582900bbd810dba72c0c37`. It is stacked on the open
hypothesis-driven Factor Research change because production and the NAS were
already running that exact commit. GitHub `main` was separately observed at
`49ad71fb...`; this migration does not pretend those branches are identical.

The legacy call chain is:

```text
build_and_publish_factor_factory
  -> publish_factor_values
       -> _load_feature_values
       -> discover_factor_specs
       -> apply_factor_semantic_lineage
       -> publish_factor_definitions
       -> _build_factor_value_frame
       -> upsert factor_value
  -> evaluate_and_publish_factor_evidence
       -> _load_factor_values + _load_market_bars
       -> build_forward_return_labels for every horizon
       -> latest cost row per symbol, then global default for missing symbols
       -> IC / Rank IC / after-cost evidence
       -> candidate derivation
       -> factor correlation
       -> replace the current as-of research rows
```

`publish_factor_values()` reads the full relevant `feature_value` history,
filters only feature set/version/timeframe and valid rows, discovers specs,
writes definitions, pivots features, computes values, performs cross-sectional
normalization, and upserts Definition and Value. Its direct side effects are the
two Gold datasets.

`evaluate_and_publish_factor_evidence()` reads the full selected Factor Value
and closed Market Bar histories, builds delayed forward labels, selects the
latest cost row per symbol, computes evidence and correlation, derives
candidates, and replaces factory-owned rows for the requested `as_of_date` in
Evidence, Candidate, and Correlation.

The old `as_of_date` is principally an output identity. It did not truncate
Feature Value or Market Bar reads. `PARITY_FULL/bootstrap_full` therefore keeps
the complete matching Feature history and seals Market Bars from the earliest
Feature timestamp through the latest Feature timestamp plus the maximum label
offset. It does not silently substitute a 30-day window. Cost selection likewise
keeps the legacy latest-row-per-symbol rule even when a historical as-of date is
requested; that bias is visible in signed `cost_snapshot` metadata and is not
silently repaired in a migration PR.

The stable Gold keys remain:

| Dataset | Primary key |
| --- | --- |
| Factor Definition | `factor_id, factor_version` |
| Factor Value | `factor_id, factor_version, symbol, timeframe, ts` |
| Factor Evidence | `as_of_date, factor_id, factor_version, timeframe, horizon_bars, decision_delay_bars` |
| Factor Candidate | `as_of_date, factor_id, factor_version, timeframe` |
| Factor Correlation | `as_of_date, factor_id_left, factor_id_right, factor_version, timeframe` |

Default specs are the twelve entries in `default_factor_registry()` whose input
features are available. The compatibility plan additionally enumerates sorted
`auto.single.<feature_name>` specs up to 200. `expression_hash` is the SHA-256
of the canonical operator semantics; formula and operator-graph hashes describe
semantic identity; `canonical_factor_id`, `duplicate_of`, correlation cluster,
and effective independence weight are recomputed by semantic lineage. The
signed plan revalidates all of those values before use.

The old empty-input behavior preserves existing Gold. The NAS result represents
that as `completed_no_update` with an explicit warning and no output datasets.
There is no empty-table publication.

Direct consumers found in the repository are Factor health/Web readers, the
daily expert export, and Alpha/Factor research bridges. They read the Gold paths
and therefore continue to see the last accepted generation while a NAS task is
pending. Import refreshes only the five relevant Lake file-index entries; it
does not rerun the whole V5 research refresh.

At the initial 2026-07-21 production audit, Factor health reported 4
definitions, 216,598 values, 16 evidence rows, 954 candidates, and 8,171
correlations. The relevant single Parquet files were approximately 684 KB for
Feature Value, 4.4 MB for Market Bar, 124 KB for Cost, and 4.6 MB for Factor
Value. These are observations, not fixed limits.

The existing shared plane was checked before adding the task: result handoff is
hidden/atomic before inbox exposure, heartbeat and lease state are separate,
the queue permission upgrader is repeatable, GC protects active snapshots, the
importer dispatches by strict task type, and the existing Entry Quality and
Alpha Factory shadow evidence had no open P0/P1 queue defect. The production
plane also contains the later `factor_research` task, so the actual strict union
now contains four task types rather than creating a second queue.

## Control and data contracts

`EffectiveFactorPlan` is a frozen, `extra="forbid"` Pydantic contract. The cloud
uses Lake file indexes plus streaming projection to read only distinct Feature
names and timestamp bounds, then serializes every Factor Spec field, semantic
hash, lineage field, causality/availability requirement, and execution parameter.
Canonical JSON produces `plan_digest`; Task and Snapshot signatures bind it to
the exact 40-character code commit.

The strict discriminated unions add:

- `FactorFactoryTask`
- `FactorFactorySnapshotManifest`
- `FactorFactoryResultManifest`
- `FactorFactoryWorkerReceipt`
- `FactorFactoryOutputDataset`
- `FactorFactoryPartitionReference`
- `FactorFactoryAntiLeakageCheck`

Existing v1 Entry Quality, Alpha Factory, and Factor Research contracts remain
valid. Factor-only fields are rejected in the other models. Supported cost
quantiles are `p50`, `p75`, and `p90`.

## Snapshot selection

Only these datasets may appear:

```text
gold/feature_value
silver/market_bar
gold/cost_bucket_daily
```

Feature rows are projected and filtered by signed Feature set/version/timeframe
and `is_valid=true`. All matching history is retained. Market rows are projected
to the label columns, restricted to the requested timeframe and closed bars,
and bounded by actual Feature timestamps plus decision delay and maximum
horizon. Costs are projected to the requested quantile and reduced to the
legacy latest row per symbol. The signed Snapshot records cost date, model,
source, selected bps, and digest. A missing symbol continues to use the existing
global default inside pure evidence computation.

Every materialized file is immutable, content-addressed in the shared blob
cache, and bound by path, SHA-256, byte count, row count, source mtime, and time
bounds. A changed source during sealing fails. Snapshot byte/row limits fail
closed; history is never silently shortened.

Only `PARITY_FULL/bootstrap_full` is implemented. The contracts bind a previous
accepted generation, but no incremental or unchanged-shard claim is made yet.
Incremental computation remains a later optimization and must satisfy plan,
hash, version, cache, and historical-integrity preconditions before activation.

## NAS computation and result

The existing `quant-research-worker` dispatches `task_type=factor_factory` only
when `QUANT_RESEARCH_FACTOR_FACTORY_ENABLED=1`. When disabled, its remote claim
operation skips Factor Factory directories, so it does not claim and fail them.
A second in-process check is fail-closed. Other task types remain claimable.

The worker verifies Task, Snapshot, signature, manifest digest, commit, plan,
input identity, and previous-generation binding. It executes only the embedded
plan through side-effect-free frame functions. It never rediscovers a factor,
derives a candidate, writes the cloud Lake, or sees an exchange secret.

Factor Value is written as Parquet shards under:

```text
outputs/factor_value/
  factor_version=<version>/timeframe=<timeframe>/date=<UTC-date>/part-*.parquet
```

Oversize day partitions are split recursively. Every shard records path,
partition identity, schema fingerprint, SHA-256, compressed bytes, rows, min/max
timestamps, exact primary key, and keyed-upsert semantics. Control outputs and
reports are separate files. The writer and cloud validator independently enforce
the Factor-specific total-result, per-shard, file-count, and uncompressed-byte
limits; the generic Entry/Alpha limit is unchanged.

Result exposure remains:

```text
local .partial -> complete local validation -> hidden remote partial
-> upload verification -> stop heartbeat -> atomic inbox rename
```

The NAS returns a definition preview, values, evidence, correlation, hard-gate
anti-leakage report, and worker report. It cannot include Factor Candidate.

## Anti-leakage and cloud validation

The worker produces more than the required 22 hard checks. They cover signed
identity and commit, exact dataset allowlist, full-history mode, immutable plan
membership, scoped/valid Feature rows, closed/scoped Market rows, exact signed
Cost records, positive delayed labels and horizons, unique input/output keys,
planned Factor membership, exact `event_time == ts`, exact availability lag,
Factor timestamp membership in the signed Feature snapshot, requested evidence
horizons and research-only decisions, correlation keys, no NAS candidate,
disabled promotion, zero live notional, and no live order effect. Only
`PASS` with zero violations can be signed and imported; WARN/FAIL/BLOCK is not
publishable.

Cloud validation recomputes signatures, all bindings, declared/actual file set,
path safety, symlink absence, compressed and uncompressed sizes, file count,
SHA-256, Parquet schemas, schema fingerprints, row counts, per-shard and global
keys, partition bounds, Factor membership, plan hashes, Definition preview,
Value/Evidence/Correlation scope, report identity, and all anti-leakage checks.
A newer desired Snapshot supersedes an older result before publication.

The cloud ignores the preview as publication authority. It rebuilds official
Factor Definition from the signed plan and derives Factor Candidate from
strictly validated evidence using the legacy decision policy. Candidate state is
limited to `KILL`, `RESEARCH`, `KEEP_SHADOW`, and `PAPER_READY`, always with
`manual_review_required=true`. Live, canary, enforce, auto-promotion, and
non-zero notional literals fail closed.

## Atomic Gold publication

The importer stages exact keyed merges for the five Gold datasets, writes a
durable transaction journal, atomically switches all dataset directories and
`gold/factor_factory_generation.json`, then validates row counts, dataset
digests, sidecars, keys, safety literals, and pointer identity. Recovery either
finishes a committed pointer or restores the complete prior set. Duplicate
imports are idempotent.

Definition and Value use keyed upsert. Evidence, Candidate, and Correlation
replace only the current as-of rows owned by `factors.factory.v0.1`, retaining
other dates and rows owned by the hypothesis-driven Factor Research plane. The
Factor Research pointer is migrated transactionally to shared ownership with
separate managed-row counts, so either generation remains independently
verifiable.

The generation pointer includes Task/Snapshot/commit/plan/input/cost identity,
Feature/Factor scope, horizon/delay/evidence parameters, previous generation,
factor membership, primary keys, per-dataset rows and hashes, publish time, and
the immutable research-only safety boundary.

## Scheduling, limits, and status

`quant-lab-factor-factory-request.timer` requests hourly at minute 38 UTC with a
bounded three-minute randomized delay. The slot was chosen from the deployed
timer inventory to avoid the hourly V5 refresh and the 15-minute Feature publish
boundaries. Request work is limited to one Polars thread, 30% CPU, 500 MB high
memory, and 900 MB hard memory. Pending requests coalesce to at most one latest
successor; a running task is never killed.

The unified status response includes the Factor plan/scope/horizon, factor and
output row counts, anti-leakage/import state, generation identity and age, worker
heartbeat/lease, bytes, duration, and peak RSS when available. An offline NAS
means `wait_no_local_fallback`.

Cloud settings:

```text
QUANT_LAB_NAS_FACTOR_FACTORY_ENABLED=0
QUANT_LAB_LOCAL_FACTOR_FACTORY_ENABLED=0
QUANT_LAB_FACTOR_FACTORY_MAX_RESULT_BYTES=2147483648
QUANT_LAB_FACTOR_FACTORY_MAX_VALUE_PARTITION_BYTES=268435456
QUANT_LAB_FACTOR_FACTORY_MAX_FILE_COUNT=20000
QUANT_LAB_FACTOR_FACTORY_MAX_UNCOMPRESSED_BYTES=17179869184
QUANT_LAB_FACTOR_FACTORY_MAX_SNAPSHOT_BYTES=26843545600
QUANT_LAB_FACTOR_FACTORY_MAX_PENDING_TASKS=1
```

NAS settings:

```text
QUANT_RESEARCH_FACTOR_FACTORY_ENABLED=0
FACTOR_FACTORY_MAX_RESULT_BYTES=2147483648
FACTOR_FACTORY_MAX_VALUE_PARTITION_BYTES=268435456
FACTOR_FACTORY_MAX_FILE_COUNT=20000
FACTOR_FACTORY_MAX_UNCOMPRESSED_BYTES=17179869184
```

The container remains the single existing worker with 3 CPUs, 8 GiB memory,
256 PIDs, read-only root, no-new-privileges, UID/GID isolation, and the shared
`/runtime/heavy-job.lock`. The reusable queue permission upgrade remains
`deploy/scripts/upgrade_research_queue_permissions.sh`; no `chmod 777`, second
SSH user, queue, cache, signature system, or container is introduced.

The legacy `qlab build-factor-factory` command remains available only when
`QUANT_LAB_LOCAL_FACTOR_FACTORY_ENABLED=1` is explicitly set for fixture,
comparison, maintenance, or manual rollback. No outage toggles it automatically.

## Acceptance and cutover

Before enabling the hourly timer:

1. Deploy the identical commit to qyun2 and the NAS image.
2. Keep the request timer disabled and local fallback disabled.
3. Enable the cloud request gate and NAS compute gate for a controlled run.
4. Run one task with `RUN_ONCE=true`.
5. Run cloud import with `--validate-only`; record signatures, anti-leakage,
   compressed/uncompressed sizes, file count, runtime, transfer/cache bytes, and
   RSS.
6. Import once, compare all five normalized tables with the old path, and verify
   Factor Research co-ownership, Web/readers, expert export, and Alpha bridge.
7. Repeat for a second consecutive shadow generation, including a warm-cache
   run. Required NAS peak RSS is below 6 GiB; cloud request must remain below
   900 MB and cloud import below 1.2 GiB.
8. Only if both shadows and all parity/resource gates pass, enable the hourly
   request timer. Keep local fallback at zero.

The V5 research refresh must not wait for the NAS. The heavy Factor command was
already absent on the production baseline used by this stacked change; cutover
therefore consists of enabling the new timer only after acceptance, not deleting
another command early.

## Manual rollback

```text
disable and stop quant-lab-factor-factory-request.timer
set QUANT_LAB_NAS_FACTOR_FACTORY_ENABLED=0
stop creating new Factor Factory tasks
allow an already claimed task to finish, or explicitly cancel it
keep the unified importer running for other task types
leave accepted Gold and generation evidence intact
set QUANT_RESEARCH_FACTOR_FACTORY_ENABLED=0 on the NAS
only in an approved maintenance window:
  set QUANT_LAB_LOCAL_FACTOR_FACTORY_ENABLED=1
  run the legacy local command manually
```

There is no automatic fallback and no change to Paper, risk, V5, canary, or
order state.

## Deferred work

Changed-shard references and Factor/Evidence compute caches are not claimed in
the bootstrap implementation. They should be added only after measured cold and
warm runs prove their keys and invalidation behavior. Candidate Labels and
Strategy Evidence are the next plausible NAS migrations, but must use separate
audits and PRs and must not inherit any Factor publication authority.
