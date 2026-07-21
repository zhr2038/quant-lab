# Factor Factory NAS Research Plane Remediation Acceptance Report

Status: **CANDIDATE ACCEPTED; PRODUCTION GO BLOCKED BY DRAFT MERGE GATE**

This report covers implementation commit
`98a76eda9c0f7b6cef68f45933c36ab343aa6693` and its report-only descendant. It
is deliberately not a final production acceptance report yet. PR #37 and PR
#38 remain Draft, GitHub `main`
is still `49ad71fb9d3043e4882546fd8b8d4ff0ba93106b`, and qyun2/NAS remain on
`ab0d4c149e8717da891b2ac11bbb03f9e783ecb9`. Evidence that can only be produced
after the final `main` merge is marked `PENDING_FINAL_MAIN` rather than being
borrowed from an older SHA.

## 1. Previous call chain

The pre-remediation request path was:

```text
create_factor_factory_task
  -> seal_factor_factory_snapshot
     -> scan Feature identities and build dated Factor Plan
     -> load Previous Generation
     -> materialize all selected Feature/Market/Cost Parquet into .sealing.partial
     -> hash the materialized files
     -> derive Snapshot ID, including as_of_date and Previous Generation
     -> only then check whether snapshots/<snapshot_id> already exists
  -> derive Task ID and enqueue NAS work
```

Consequences were full cloud materialization before duplicate detection, a new
Snapshot identity when output date or Previous Generation changed, and failure
when an existing Snapshot retained its manifest but GC had released `files/`.

## 2. Previous Snapshot identity

The v1 Snapshot seed included:

```text
quant_lab_commit
as_of_date
factor_plan_digest (whose plan included created_at)
materialized source_input_digest
materialized cost_input_digest
previous_generation_id
previous_generation_digest
previous_generation_manifest
horizons / delay / samples / quantiles / result mode / history mode
```

The v1 implementation created `.sealing.*.partial` and all projected Parquet
before it could calculate this seed. Existing-Snapshot verification required
`files/`, so `FILES_RELEASED.json` was not a reusable state.

## 3. New Snapshot identity

Factor Snapshot v2 identity is calculated before payload creation from exactly:

```text
quant_lab_commit
factor_plan_digest
source_input_digest
cost_input_digest
feature_set / feature_version / factor_version / timeframe
horizon_bars / decision_delay_bars / max_factors / min_samples
top_quantile / cost_quantile / result_mode / history_mode
```

`as_of_date`, `generated_at`, and all Previous Generation fields are excluded.
Factor Plan v2 excludes audit-only `created_at` from `plan_digest`. Snapshot v1
continues to parse and verify with its legacy digest/signature reconstruction;
it is not silently rewritten into v2.

## 4. Task identity

The Task remains the concurrency and publication binding. Its identity includes:

```text
snapshot_id
factor/source/cost digests
previous_generation_id / previous_generation_digest
task parameters, including output as_of_date
quant_lab_commit
signing key ID
```

Changing Previous Generation changes Task ID/publication binding but not
Snapshot ID. V2 worker/result validation therefore compares task-only date and
Previous Generation at the Task/Result boundary, while retaining v1 checks.

## 5. Lake File Index v0.3

The index now records:

```text
dataset / path / size_bytes / mtime_ns / sha256
row_count / min_ts / max_ts / schema_fingerprint
uncompressed_bytes / indexed_at / index_version
```

Real SHA-256 and Parquet metadata are reused only while size and mtime are
unchanged. New/changed files alone are re-read. A source changing during index
construction fails closed. Replacement is atomic, and diagnostic reuse state
does not cause a second whole-index rewrite. Final Snapshot identity uses source
SHA plus deterministic projection/filter rules rather than path+mtime alone.

## 6. No-Change fast path

The new request order is:

```text
refresh/read File Index
-> calculate selected source/cost/plan identity
-> derive candidate Snapshot ID
-> compare current Generation and no-update pointers
-> enforce minimum recompute interval
-> only then materialize or rehydrate
```

An exact current match returns `already_current`, `task_created=false`, and
`snapshot_materialized=false`. A matching empty-input pointer returns
`already_current_no_update` with the same no-work guarantees. The comparison
does not use date. CLI/systemd emits explicit success events and exits zero.
Status reports `up_to_date` rather than failure or permanent Pending.

## 7. Released Snapshot rehydrate state machine

```text
manifest + SEALED + FILES_RELEASED, no files/
-> acquire per-Snapshot process and OS advisory lock
-> verify retained digest, signature, key ID, seal, identity, and capacity
-> rebuild into .rehydrate.<snapshot>.<uuid>.partial/files
-> compare every path/SHA/size/row/schema/min/max reference exactly
-> atomically install files/
-> remove FILES_RELEASED.json
-> verify full Snapshot again
-> restore 0550/0440 permissions and append audit evidence
```

Any source, plan, commit, cost selection, file set, projected content, schema,
row, or bound mismatch returns `snapshot_rehydrate_identity_mismatch`; the old
manifest/signature is never changed. Concurrent callers serialize on the same
lock. A later request repairs the narrow crash window after payload install,
and stale partials are cleaned only while their Snapshot lock is available.

## 8. Snapshot GC compatibility

Pending, Running, Inbox, and active rehydrate Snapshots remain protected.
Completed payloads may be released while manifest, signature, seal, and audit
remain. `FILES_RELEASED.json` is legal. GC and rehydrate use the same lock, so a
payload cannot be released during restoration. Stale `.rehydrate.*.partial`
cleanup is independent from active work.

## 9. Empty no-update handling

An accepted `completed_no_update` result atomically records
`gold/factor_factory_no_update_state.json` with Snapshot/plan/source/cost/commit
and full parameter identity. Gold remains unchanged. Repeating the same empty
identity skips rehydrate and NAS work. New input changes identity and permits a
new Task.

## 10. NAS capacity gates

Factor-specific defaults are now:

| Gate | Default |
| --- | ---: |
| Compressed Snapshot | 2 GiB |
| Estimated uncompressed input | 4 GiB |
| Compressed result | 512 MiB |
| Value partition | 128 MiB |
| Uncompressed result | 1 GiB |
| Result files | 20,000 |

Snapshot v2 carries total and Feature/Market/Cost uncompressed estimates from
Parquet metadata. The NAS checks the 2 GiB compressed limit before transfer and
the 4 GiB estimate before compute. A legacy v1 payload derives its uncompressed
size from local Parquet metadata before any full DataFrame read. The compute is
still monolithic; this is a fail-closed 8 GiB-container ceiling, not a claim of
20–25 GiB support. Final live acceptance still requires peak RSS below 6 GiB.

## 11. Importer capacity gates

Importer ordering is signature/binding, declared file count/set, file SHA/size,
Parquet metadata, per-partition limit, compressed/uncompressed totals, capacity
gates, then row/key/global scans. Defaults are 512 MiB compressed and 1 GiB
uncompressed under `MemoryMax=3G`.

Gold merge and global primary-key verification use DuckDB with two threads,
`memory_limit='768MB'`, disabled insertion-order preservation, and a transaction
local spill directory. A synchronous write probe fails fast if spill is not
writable; cleanup runs on success and failure. Final live acceptance still
requires Importer peak RSS below 3 GiB and measured spill peak.

## 12. Changed-Shard decision

Changed-Shard is deferred to an independent PR. `PARITY_FULL/bootstrap_full`
and all Factor math remain unchanged. The hourly timer is an identity poll, not
an unconditional full run; exact No-Change runs do no payload/NAS work, while
changed input has a six-hour minimum recompute interval (at most four full Tasks
per UTC day).

The retained 2026-07-21 production sample contained seven Factor Snapshots and
six observed input-identity changes; payload average was 1,250,943 bytes and
maximum 1,252,162 bytes. This is a partial one-day sample.

The independent PR becomes required before shortening the six-hour floor, or
when any threshold persists for seven days: more than four changed identities
per day, full-run p95 over 20 minutes, NAS peak over 5 GiB, compressed Snapshot
over 512 MiB, or result transfer over 256 MiB. Acceptance must bind unchanged
references to exact previous Generation/partition SHA/rows/bounds, match two
consecutive `PARITY_FULL` shadows, avoid at least 80% unchanged transfer bytes,
and retain 40/40 Anti-Leakage PASS.

## 13. Modified files

Candidate `98a76ed` changes 26 files:

- deployment limits and scheduling under `deploy/nas_research_worker` and
  `deploy/systemd`;
- runbook and explicit `pyarrow` dependency;
- CLI, File Index, Factor Plan, Research contracts, Snapshot/Task queue,
  Rehydrate/GC/status, Result/Importer/Gold publication, worker/runner/writer;
- new `src/quant_lab/research_plane/snapshot_lock.py`;
- Factor Research Plane, File Index, and systemd regression tests.

No Factor formula/member, Alpha rule, Paper/risk/V5/live behavior, AI prompt,
Candidate threshold, horizon, delay, cost formula, or local fallback changed.

## 14. Added regression coverage

Coverage includes:

- same input/date independence and Plan/source/cost/parameter identity changes;
- Previous Generation changes Task but not Snapshot;
- pre-materialization No-Change and CLI exit zero;
- minimum interval deferral;
- strict release/rehydrate identity, signature preservation, concurrency,
  crash recovery, source mismatch, stale cleanup, and GC exclusion;
- Empty No-Update repeat and later input appearance;
- v2 estimate and legacy v1 metadata input gates before read;
- result total/partition/file gates before global scans;
- bounded DuckDB configuration and writable spill path;
- File Index real SHA/schema/uncompressed metadata/reuse/mutation failure;
- systemd/NAS limits and no local fallback.

## 15. Local validation

| Check | Result |
| --- | --- |
| Full pytest | `1551 passed, 4 skipped in 268.66s` |
| Cross-plane focused pytest | `177 passed, 1 skipped` |
| Factor/GC focused pytest | PASS |
| `ruff check .` | PASS |
| `python -m compileall -q src deploy tools` | PASS |
| `git diff --check` | PASS |
| Frontend TypeScript/Vite build | PASS, 2,881 modules |
| NAS Docker Compose config | PASS with Compose v5.1.3 |

## 16. GitHub CI

Candidate `98a76ed` has two successful CI runs at the exact SHA:

- push run `29847349061`: PASS in 4m24s;
- pull_request run `29847351446`: PASS in 4m28s.

Each workflow installed `.[dev]`, ran full Ruff, and ran full pytest.

## 17. PR #37

- Head: `6652c3669f09ad25ce582900bbd810dba72c0c37`.
- Base: `main`.
- GitHub: `OPEN`, `CLEAN`, `MERGEABLE`, both CI runs SUCCESS.
- Review threads/reviews: none.
- Code/safety review: no P0/P1 found; signed Research Plane, atomic cloud
  authority, zero-live-effect boundaries, and exact generation binding remain.
- Draft decision: keep Draft until the owner explicitly authorizes merge; its
  own PR text says it must not be auto-merged.

## 18. PR #38 topology

- Head: `98a76eda9c0f7b6cef68f45933c36ab343aa6693`.
- Current base: `refactor/hypothesis-driven-factor-research`.
- GitHub: `OPEN`, `CLEAN`, `MERGEABLE`, Draft, both CI runs SUCCESS.
- Direct-main rebase: `PENDING_FINAL_MAIN`; it cannot be performed correctly
  until PR #37 is merged.

## 19. Final candidate SHA

Reviewed implementation commit:
`98a76eda9c0f7b6cef68f45933c36ab343aa6693`. The PR may have a later
documentation-only descendant containing this report. Neither is yet the final
production SHA because the required main rebase and merge have not occurred.

## 20. qyun2/NAS deployment SHA

Current verified state on 2026-07-22:

| Surface | SHA / state |
| --- | --- |
| GitHub main | `49ad71fb9d3043e4882546fd8b8d4ff0ba93106b` |
| qyun2 repo | clean `ab0d4c149e8717da891b2ac11bbb03f9e783ecb9` |
| NAS repo | clean `ab0d4c149e8717da891b2ac11bbb03f9e783ecb9` |
| NAS image worker commit | `ab0d4c149e8717da891b2ac11bbb03f9e783ecb9` |
| qyun2 Factor timer | disabled / inactive |
| Factor pending / running | 0 / 0 |
| unified Importer timer | active |
| NAS worker | running / healthy, `RUN_ONCE=false` |

Final-main deployment: `PENDING_FINAL_MAIN`.

## 21. Production No-Change measurement

`PENDING_FINAL_MAIN`. Required evidence is two consecutive final-main requests;
the second must return `already_current`, `task_created=false`, and
`snapshot_materialized=false`, with wall time, peak RSS, and process I/O bytes.
Old-SHA observations are not accepted as evidence for this candidate.

## 22. Production Rehydrate measurement

`PENDING_FINAL_MAIN`. Required evidence is release of a final-main v2 payload,
same-identity request, unchanged manifest/signature/file references, successful
NAS claim, and no residual release/partial marker.

## 23. Final RUN_ONCE

`PENDING_FINAL_MAIN`. Required chain:

```text
RUN_ONCE -> NAS Compute -> 40/40 PASS -> strict cloud validation
-> five-Gold atomic publication -> Generation verify -> Snapshot release
```

## 24. Final Shadow

`PENDING_FINAL_MAIN`. One further final-SHA Shadow must pass after RUN_ONCE.

## 25. Resource evidence

| Stage | Current evidence | Final-main requirement |
| --- | --- | --- |
| No-Change request | not yet measured | no materialization; wall/RSS/I/O |
| Changed request | old v1 payload about 1.25 MB | wall/RSS/I/O/temp peak |
| NAS | prior shadow peak 605,286,400 B | input bytes/estimate, rows, wall, RSS <6 GiB |
| Importer | historical cgroup peak unavailable | result bytes, wall, RSS <3 GiB, spill peak |

Missing metrics remain explicitly unknown until final-main execution.

## 26. Remaining risks

1. PR #37 and #38 are still unmerged Drafts; production cannot truthfully claim
   main provenance.
2. Worker compute remains monolithic. The 4 GiB gate is conservative but final
   resource proof applies only to the measured production input.
3. Changed-Shard is not implemented; the six-hour floor limits, but does not
   optimize, changed full-history runs.
4. Only a partial one-day input-change sample is available; retain 14 days of v2
   request identity evidence.
5. Importer and request historical oneshot cgroups did not retain full peak/I/O
   counters, so final acceptance must measure them explicitly.

## 27. Rollback

Before final cutover, rollback is simply to leave the Factor request timer
disabled; no candidate code is running in production.

After final cutover:

```text
disable and stop quant-lab-factor-factory-request.timer
set QUANT_LAB_NAS_FACTOR_FACTORY_ENABLED=0
set QUANT_RESEARCH_FACTOR_FACTORY_ENABLED=0 on NAS
do not enable QUANT_LAB_LOCAL_FACTOR_FACTORY_ENABLED automatically
allow or explicitly cancel an already claimed signed task
keep unified Importer available for other task types
retain accepted Gold, generation pointer, manifests, signatures, and audit
redeploy the previous known-good main SHA only with an explicit rollback decision
```

## Twelve-condition verdict

| # | Condition | Candidate status |
| ---: | --- | --- |
| 1 | Snapshot/Previous Generation decoupled | PASS |
| 2 | No-Change before materialization | PASS |
| 3 | Strict Released Snapshot rehydrate | PASS |
| 4 | Empty No-Update repeat | PASS |
| 5 | GC/Rehydrate compatibility | PASS |
| 6 | Credible NAS input gates | PASS_CODE; live RSS pending |
| 7 | Importer gate before global scan | PASS |
| 8 | Full tests | PASS |
| 9 | PR #37 mergeable or merged | PASS_MERGEABLE; still Draft |
| 10 | PR #38 directly based on main | **FAIL_PENDING_MERGE** |
| 11 | final main equals production/NAS/task/result | **FAIL_PENDING_MERGE** |
| 12 | final SHA RUN_ONCE plus Shadow | **FAIL_PENDING_DEPLOYMENT** |

Overall verdict: **NO-GO until conditions 10–12 are proven on final main**.
