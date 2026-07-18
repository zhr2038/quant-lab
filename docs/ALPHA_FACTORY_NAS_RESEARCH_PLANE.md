# Alpha Factory NAS Research Compute Plane

## 1. Scope and invariants

This migration moves only the historical compute behind `qlab build-alpha-factory`,
including Second Stage Alpha Factory, to the existing NAS Research Compute Plane.
It does not modify V5, exchange execution, Risk Permission, Paper ACK/Tracker,
Paper Promotion Gate, current Entry Quality, Factor Factory, Cost Calibration,
Lake Compaction, AI Research, or real-time market collection.

Every Alpha Factory task, snapshot, result, receipt, publication pointer, and Web
status is constrained to:

```text
research_only=true
live_order_effect=none
automatic_promotion=false
max_live_notional_usdt=0
```

The NAS has no exchange credentials and cannot write cloud Lake or Gold. It cannot
publish Promotion Queue, merge Strategy Evidence, create Paper Trackers, change
Risk Permission, enable Canary, or submit an order. Production remains `shadow`.
There is no automatic local fallback when the NAS is unavailable.

## 2. Legacy side-effect audit

The legacy call chain was:

```text
quant-lab-alpha-factory.timer (03:20 UTC)
  -> quant-lab-alpha-factory.service
  -> qlab build-alpha-factory
  -> build_and_publish_alpha_factory
  -> publish_alpha_factory_template_registry
  -> build_and_publish_second_stage_alpha_factory
  -> read historical Lake inputs
  -> build Factor Bridge, Candidate, Result, Promotion Queue
  -> replace dedicated Gold datasets
  -> merge shared Strategy Evidence datasets
```

Second Stage previously wrote its sample, summary, relative-strength, and
exit-policy datasets before the outer Alpha Factory publish completed. The outer
function then wrote Candidate, Result, Promotion Queue, and Strategy Evidence. A
failure could therefore leave a mix of new and old generations.

Factor Bridge also preferred the newest cloud `quant_lab_expert_pack_*.zip` and
only recomputed from Lake under a cloud row limit. That dependency is removed.
The manual legacy path now calls direct Lake recompute too, so Expert Pack export
can consume Alpha output without Alpha compute depending on Expert Pack export.

## 3. Migrated flow and boundary

```text
cloud request timer
  -> prepare effective Template Registry
  -> build/refresh bounded Lake File Index
  -> seal projected immutable Alpha Snapshot
  -> sign deterministic AlphaFactoryTask
  -> research_queue/pending

NAS quant-research-worker
  -> claim + lease
  -> verify Task and Snapshot signatures
  -> content-addressed Blob sync
  -> compute Second Stage Alpha Factory
  -> recompute Factor Forward Validation and Factor Bridge
  -> compute Alpha Candidate and Alpha Result
  -> run fail-closed Anti-Leakage checks
  -> sign Result and Receipt
  -> hidden partial upload
  -> stop heartbeat
  -> atomic inbox handoff

cloud shared importer
  -> verify every binding, file, schema, row, and safety invariant
  -> derive Promotion Queue
  -> derive managed Strategy Evidence deltas
  -> merge only the current Alpha-managed window
  -> durable atomic multi-dataset publish
  -> generation pointer
  -> completed/imported state
```

Pure compute entry points are `compute_second_stage_alpha_factory`,
`compute_factor_bridge_source`, `compute_alpha_factory`, and
`compute_alpha_factory_from_snapshot`. Cloud-only control/publication entry points
are `prepare_alpha_factory_control_state`, `derive_alpha_factory_cloud_outputs`,
`merge_alpha_factory_managed_evidence`, and `publish_alpha_factory_generation`.

## 4. Phase 0 Research Plane repairs

The shared blockers were fixed before adding `task_type=alpha_factory`:

1. Result handoff uploads a hidden partial, writes a completeness marker, stops
   heartbeat, exposes the inbox directory atomically, and then transfers state
   ownership to cloud.
2. Heartbeats write `lease/<task_id>.json`; they do not rewrite business status.
   Lease sequence is monotonic.
3. Lease recovery exposes complete hidden results, leaves visible inbox results
   cloud-owned, and requeues only when no complete result exists.
4. SSH and SCP have bounded timeouts. Failed uploads remove their remote partial
   path best-effort.
5. Queue upgrade creates `quant-research` before tmpfiles and recursively repairs
   existing queue ownership and modes without `chmod 777`.
6. The writable File Index path is
   `/var/lib/quant-lab/lake/bronze/lake_file_index`.
7. Cloud and NAS examples use `cloud-research-v1` and `nas-research-v1`.
8. Candidate anti-leakage symbol lookup accepts `symbol` or `normalized_symbol`.
9. Validation failure rejects a result; publication infrastructure failure keeps
   a valid result in inbox as `publish_retry_pending`.
10. Snapshot GC keeps signed manifests and receipts while releasing payload files.
11. Entry Quality `recent_7d` normalization happens before Task signing.
12. Row-level commit and `source_version` provenance are strict and non-null.
13. Result size is capped at 256 MiB, consistent with importer memory limits.
14. Durable publication journals recover interrupted directory exchange.

## 5. Shared multi-task contract

The queue, Blob cache, SSH identity, worker, heavy-job lock, importer, lease,
heartbeat, and Snapshot GC are shared. No `alpha_queue` or second worker exists.

Strict Pydantic discriminated models cover Entry Quality and Alpha Task, Snapshot,
Result, and Receipt variants. All use `extra="forbid"`; Alpha fields cannot enter
an Entry Quality v1 Task, while the existing Entry Quality wire contract remains
compatible.

Alpha Task identity is deterministic over Snapshot ID, date, lookback, candidate
cap, Registry digest, full commit, and signing Key ID. The Task binds a full
40-character commit, Snapshot digest, Registry digest, selected V5 bundle, Alpha
schema, Second Stage schema, and safe execution invariants. Lookback is limited to
1-180 days and candidates to 1-200.

## 6. Effective Template Registry

Cloud reads the existing registry, merges code defaults, disables unknown
templates, validates known templates as `paper_shadow_only`, and requires the only
live-notional option to be zero. It seals the effective registry and signs its
digest into Task and Result.

The NAS can read but cannot modify or publish it. Import recomputes the current
effective digest. A result made against an older registry is superseded/rejected
and cannot overwrite newer control state.

## 7. Snapshot input inventory

| Dataset | Projection and window semantics |
| --- | --- |
| `silver/market_bar` | projected OHLCV/closure fields; selected symbols; lookback plus max pre-window and label horizon |
| `gold/expanded_universe_quality` | latest effective date at or before task date only |
| `gold/cost_bucket_daily` | involved symbols and bounded lookback |
| `gold/market_regime_daily` | bounded historical and current regime state |
| `gold/btc_probe_exit_policy_review` | bounded entry/exit/MFE/MAE fields |
| `gold/paper_strategy_runs` | bounded Paper/exit evidence |
| `gold/strategy_evidence` | current task-day control summary |
| `gold/strategy_evidence_sample` | bounded historical samples |
| `gold/factor_candidate` | bounded candidate identity/family fields |
| `gold/factor_value` | selected symbols, factors, values, and availability timestamps |
| `gold/alpha_factory_template_registry` | cloud-generated effective control registry |

Maximum horizon comes from `DEFAULT_HORIZONS`; it is not duplicated. File
membership comes from the refreshed Lake File Index, then each file is scanned
lazily with time, symbol, and column projection. Snapshots are immutable and
content-addressed by source identity, control digest, parameters, and projections.
Input byte and row caps are enforced before Task publication.

## 8. Second Stage and Factor Bridge

The same formulas and decisions are preserved:

* Expanded relative strength ranks only information available at Decision Time.
* Futures/spot inverse proxy remains research-only, has neither futures data nor
  funding, and can never become `PAPER_READY`.
* Exit Policy Review compares actual exit with fixed 4/8/12/24/48 hour holds and
  carries MFE, MAE, net delta, and incomplete-horizon semantics.
* Pair/market-neutral samples remain shadow evidence.

Market bars are projected once, sorted once by symbol/time, and indexed once.
Lazy scans use streaming collection; full market history is not converted to an
unbounded Python dictionary.

Factor Bridge is recomputed from Snapshot `factor_candidate`, `factor_value`,
`market_bar`, `market_regime_daily`, and `cost_bucket_daily`. Cloud Expert Pack
ZIPs are not read. The compatibility CSV remains in the Result for audit/export.
Every bridge row is `RESEARCH`, `strategy_review_only=true`, and has no live effect.

## 9. Candidate, Result, and promotion safety

Candidate and Result preserve existing formulas, stable Candidate IDs,
chronological 70/30 train/validation split, and task-end recent-seven-day window.
The explicit cap remains 200.

Allowed Result decisions are only `RESEARCH`, `KEEP_SHADOW`, `KILL`, and
`PAPER_READY`. `LIVE_SMALL_READY`, `LIVE`, `CANARY`, `ENFORCE`, `AUTO_PROMOTE`,
unknown states, nonzero notional, unsafe safety mode, Paper-ready futures proxies,
or promotional Factor Bridge rows reject the whole bundle.

`PAPER_READY` remains a research output only. NAS does not publish it to Paper.
Cloud Promotion Queue always requires manual live approval and does not bypass
existing Paper gates.

## 10. Output datasets and keys

| Dataset | Primary key | Empty result |
| --- | --- | --- |
| `second_stage_alpha_factory_sample` | `as_of_date,strategy_candidate,symbol,source_event_key,horizon_hours` | clear task window |
| `second_stage_alpha_factory_summary` | `as_of_date,strategy_candidate,symbol,regime_state,horizon_hours` | clear task window |
| `expanded_relative_strength_decision_sample` | `decision_ts,symbol,lookback_hours,top_k,selected_rank,label_horizon_hours` | clear task window |
| `exit_policy_review_sample` | `as_of_date,strategy_id,source_entry_id` | clear task window |
| `exit_policy_review_summary` | `as_of_date,strategy_id,symbol` | clear task window |
| `alpha_factory_candidate` | `as_of_date,candidate_id` | clear task window |
| `alpha_factory_result` | `as_of_date,candidate_id` | clear task window |
| `alpha_factory_promotion_queue` | cloud-derived from validated Result | clear task window |

Template Registry stays cloud-managed. Shared `strategy_evidence_sample` and
`strategy_evidence` publication replaces only the task date's Alpha-managed rows
and preserves every other producer/date. A valid empty result removes stale Alpha
rows for that date without clearing unrelated evidence.

## 11. Result and cloud validation

NAS returns seven Parquet outputs plus:

* `reports/factor_strategy_bridge_candidates.csv`
* `reports/alpha_factory_worker_report.json`
* `reports/alpha_factory_anti_leakage.json`

Manifest/Receipt bind Task, Snapshot ID/SHA, cloud/worker commits, Registry digest,
V5 bundle ID, parameters, file SHA/schema/rows, input/cache/download bytes, peak
RSS, duration, and Anti-Leakage.

Cloud validates all signatures and Key IDs; full commits; binding digests; exact
output/report sets; path/symlink safety; file size/hash; Parquet schema/fingerprint;
row count/scope/limits; primary-key uniqueness; Candidate/Result one-to-one
identity; decision/safety whitelists; current Registry; current commit; and
supersession.

## 12. Anti-Leakage gate

Publication requires every check to be `PASS` with `violation_count=0`:

1. relative-strength rank uses only pre-decision bars;
2. future labels are absent from ranking features;
3. Decision Bar equals an actual completed target bar;
4. label horizon is complete and inside Snapshot bounds;
5. train/validation is chronological 70/30;
6. recent window is exactly the final seven days;
7. quality, regime, and cost are not from the future;
8. Factor Forward Validation honors Decision Delay;
9. Factor Bridge binds the same Snapshot inputs;
10. Expert Pack cache is not authoritative;
11. source dataset, commit, and Registry bindings agree;
12. no live action exists.

`WARN`, `FAIL`, `BLOCK`, a missing check, or mismatched check count rejects the
entire Result.

## 13. Cloud derivation and durable publish

Importer ignores any NAS attempt to provide Promotion Queue or shared Strategy
Evidence. Cloud derives them only after strict validation. Existing consumers
continue reading the same Gold names and schemas: Expert Pack, Web/API, System
Acceptance, Alpha views, and opportunity metadata.

`commit_atomic_research_generation` stages every dedicated/shared output, writes
a durable journal, swaps datasets, writes/verifies the generation pointer, and
then cleans backups. Restart recovery completes a verified generation or restores
backups. Duplicate import is idempotent. Temporary publication failure leaves the
valid inbox result retryable.

`gold/alpha_factory_generation.json` records generation, task, Snapshot, commit,
Registry digest, date, row counts, publish time, and non-live invariants.

## 14. Services, Docker, permissions, and retention

The new request timer keeps 03:20 UTC plus 20-minute randomized delay. It only
prepares control state, Snapshot, and Task:

```text
CPUQuota=30%
MemoryHigh=500M
MemoryMax=900M
POLARS_MAX_THREADS=1
```

Shared importer remains capped at 1.2 GiB and 256 MiB result bundles. The old
Alpha service is retained as disabled manual fallback behind
`QUANT_LAB_LOCAL_ALPHA_FACTORY_ENABLED=1`; NAS failure never enables it.

The existing NAS worker uses the shared heavy-job lock:

```text
cpus=3.0
mem_limit=8g
pids_limit=256
read_only=true
no-new-privileges=true
```

Observed worker peak RSS must stay below 6 GiB.

The queue permission migration creates user/group before tmpfiles and applies
`2770` directories plus `0660` metadata without world-write. Completed manifests,
signatures, receipts, validation, and audit remain. Snapshot GC releases payloads
after successful import and enforces retention/capacity with audit records.

## 15. Web visibility

Research Compute status exposes separate `entry_quality_history` and
`alpha_factory` states. Web shows Task/Snapshot, input/download/cache bytes, peak
RSS, duration, result rows, Anti-Leakage, and last error, plus `research-only`,
`live_order_effect: none`, and `wait_no_local_fallback`.

## 16. Local verification evidence

| Check | Result |
| --- | --- |
| Phase 0 + Alpha + Second Stage + systemd tests | 112 passed |
| Full pytest | 1413 passed, 3 skipped |
| Ruff | passed |
| compileall | passed |
| git diff --check | passed |
| frontend production build | passed |
| Compose config on Windows | unavailable because Docker CLI is not installed; mandatory on NAS |

The fixed old-vs-new fixture compares all seven compute outputs, Factor Bridge,
cloud-derived Promotion Queue, and managed Strategy Evidence after normalizing
timestamps, host paths, and generation IDs. Numeric values, decisions, reasons,
Candidate IDs, and schemas match.

## 17. Performance acceptance record

Production values must be measured during shadow deployment, not inferred from
unit fixtures.

| Metric | Legacy cloud | Snapshot/cloud | NAS cold | NAS warm | Cloud import |
| --- | ---: | ---: | ---: | ---: | ---: |
| wall time | pending | pending | pending | pending | pending |
| peak RSS | pending | pending | pending | pending | pending |
| read/write bytes | pending | pending | input/result bytes | cache/download bytes | result bytes |

Worker report and Result manifest already capture input, cache hit, download,
result size, duration, peak RSS, Candidate/Result counts, decisions, and
Anti-Leakage. Two shadow runs must populate this table before timer enablement.

## 18. Deployment sequence

1. Install Phase 0 code and permission migration only.
2. Run Entry Quality Research Plane regression.
3. Install request service/timer but leave timer disabled.
4. Build NAS worker with exact deployed full commit.
5. Run `docker compose config` on NAS.
6. Run one Alpha Task with `RUN_ONCE=true`.
7. Run cloud importer in validate-only mode.
8. Compare all old/new outputs.
9. Perform one formal import and verify generation pointer.
10. Run two Alpha Factory shadow generations.
11. Verify Expert Pack and Web read the new Gold generation.
12. Disable legacy local timer.
13. Enable the request timer.

No new timer may be enabled before Phase 0 and shadow validation pass.

## 19. Manual rollback

Stop/disable request timer, stop dispatch, let claimed work finish or cancel it,
leave the shared importer available to other tasks, set
`QUANT_LAB_NAS_ALPHA_FACTORY_ENABLED=0`, and use local legacy service only in an
approved maintenance window with `QUANT_LAB_LOCAL_ALPHA_FACTORY_ENABLED=1`.

Do not delete queue history, Snapshot manifests, receipts, validation, or Gold
journals. Do not automatically send the 4 GiB workload back to qyun2.

## 20. Remaining deployment risks

* Measure actual qyun2 legacy/Snapshot/import duration, RSS, and bytes.
* Measure NAS cold/warm duration, peak RSS, and cache ratio.
* Pass Compose expansion and image build on NAS.
* Pass one validate-only and two imported shadow runs against production Gold.
* Confirm real Snapshot input remains below limits.
* Inspect live Web against the deployed API.

These are acceptance items, not reasons to relax safety or silently fall back.

## 21. Next migration recommendation

Only after stable Alpha shadow evidence should V5 Research Refresh be audited for
a similar compute/publish split. It must stay separate from V5 execution, Paper
Tracker state, Risk Permission, and live order behavior.
