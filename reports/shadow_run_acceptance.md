# Shadow Run Acceptance

## Verdict

`PASS_FACTOR_DISCOVERY_V2_RESEARCH_SHADOW`

This verdict is limited to signed Research Plane execution, strict cloud
validation, atomic Gold publication, exact Factor-to-Alpha generation binding,
and zero live effect. It is not evidence of profitability, Paper readiness,
Canary readiness, or permission to restore production alpha.

## Identities

- Cloud commit: `8d8c884ad9fc35ff600921fc8c0a05bf732cff72`
- NAS Worker commit: `8d8c884ad9fc35ff600921fc8c0a05bf732cff72`
- Branch: `refactor/hypothesis-driven-factor-research`
- V5 read-only commit: `3df1c67cc44cc8be364ec5d3798ea0d3595c0abc`

## Accepted shadow tasks

| Metric | Factor Research | Bound Alpha Factory |
| --- | ---: | ---: |
| Task ID | `factor-research-097a452bfeb6edfd8baf5d1e` | `alpha-factory-c3b0425791b3ce7800ed1470` |
| Snapshot ID | `factor-research-b7ba4acfe2d475c65fa0d474` | `alpha-factory-d6cdd2d63e20ce52a39c8183` |
| Hypotheses | 2 active | exact Factor generation binding |
| Trial count | 8 | n/a |
| Input bytes | 388,332 | 16,295,243 |
| Downloaded bytes | 311,072 | 16,220,714 |
| Cache hit bytes | 77,260 | 74,529 |
| Cache hit ratio | 19.90% | 0.46% |
| Compute duration | 24.88 s | 20.65 s |
| Peak RSS | 354,426,880 B | 1,439,264,768 B |
| Output rows | 22,122 | 307,138 |
| Result bytes | 554,306 | 2,562,375 |
| Cloud import wall time | 4.06 s | 120.68 s |
| Cloud import CPU time | 1.65 s | 17.99 s |
| Anti-Leakage | PASS | PASS |
| Formal import | PASS | PASS |

Both NAS peaks are below the 6 GiB acceptance ceiling. Both result bundles
passed signature, task, snapshot, commit, schema, hash, row-count, safety, and
all-PASS Anti-Leakage validation before publication.

## Published Factor generation

- Generation ID: `factor-research-097a452bfeb6edfd8baf5d1e`
- Generation digest:
  `82655622935aa4fcfc6e153162ff4c4edb7450b41f098b8cfdf8043eb5511f52`
- Registered hypotheses: 4 independent, 2 active, 2 data blocked.
- Current trials: 8 confirmatory trials.
- Trial decisions: 8 `REJECTED_DATA_QUALITY`.
- FDR passes: 0.
- `SIGNAL_VALID`: 0.
- `PORTFOLIO_FAIL`: 0 because no trial reached portfolio qualification.
- `PAPER_CANDIDATE`: 0.
- Point-in-time cost coverage: 74.7907%, below the 80% gate.
- Attribution rows: 8; null primary keys: 0.
- Portfolio validation rows: 8; null primary keys: 0.

Rejecting all eight trials is the correct fail-closed outcome. Market-breadth
feature coverage is high, but point-in-time cost coverage is insufficient.
Low-volatility variants additionally have inadequate feature coverage. No
threshold was lowered to manufacture a passing alpha.

## Bound Alpha generation

Alpha generation `alpha-factory-c3b0425791b3ce7800ed1470` binds the exact
Factor generation ID and digest above. Its 165 result and promotion rows are:

- 65 `KEEP_SHADOW`
- 78 `KILL`
- 22 `RESEARCH`
- 0 `PAPER_READY`, live, canary, or enforce decisions
- `max_live_notional_usdt=0` for every row
- `manual_live_approval_required=true` for every promotion row

## Idempotency and publication

- Validate-only did not publish Gold.
- Formal imports moved each task to `completed` exactly once.
- Current generation pointers bind exact task, snapshot, commit, and digests.
- Alpha cannot consume an unversioned latest `factor_candidate` table.
- Null-key legacy placeholders are removed only from the managed Factor tables;
  unrelated generations and shared evidence rows are retained.
- The shared Research Queue and shared Worker were reused. No second queue,
  worker, key hierarchy, or cloud fallback was created.

## Safety

V5 remained clean at its original commit. No real order, position, target,
capital, Paper ACK/Tracker, risk permission, canary, or enforce state was
changed. All accepted records keep `research_only=true`,
`live_order_effect=none`, `automatic_promotion=false`, and zero live notional.

## Final regression

- Full pytest: `1494 passed, 4 skipped in 283.48s`.
- Ruff: pass.
- `compileall` for `src`, `deploy`, and `tools`: pass.
- Frontend TypeScript and Vite production build: pass.
- NAS `docker compose config --quiet`: pass.
- `git diff --check`: pass.

The local Windows environment has no Docker CLI. Compose validation was run on
the actual NAS host that built and executed the accepted image.
