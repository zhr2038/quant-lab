# V5 Decision Receipt — read-only integration design

Status: **DESIGN ONLY / NOT DEPLOYED / DEFAULT OFF**  
Audit scope: V5-prod commit `3df1c67cc44cc8be364ec5d3798ea0d3595c0abc`  
Schema: `schemas/v5_decision_receipt.schema.json`

## Objective and non-goals

The receipt is an append-only explanation of one V5 decision cycle. It must be
possible to reconstruct why the observed target weights and order intents were
produced without reading exchange secrets or mutating trading state. It is not
an order source, a risk override, a production gate, or an Alpha-enablement
mechanism.

The current `src/reporting/decision_audit.py::DecisionAudit` captures useful
scores, target weights, router decisions, counts and notes, but it is rewritten
within a run and does not bind all inputs to immutable content, code, config,
container and market-data identities. It therefore remains an input to the
receipt rather than becoming the receipt itself.

## Default-off contract

Proposed configuration (no production configuration is changed by Audit v2.1):

```text
V5_DECISION_RECEIPT_ENABLED=0
V5_DECISION_RECEIPT_MODE=read_only
V5_DECISION_RECEIPT_DIR=reports/decision_receipts
V5_DECISION_RECEIPT_MAX_INLINE_BYTES=65536
```

Enabling the writer must not alter universe selection, Alpha, portfolio,
preflight, Quant-Lab permission, risk permission, arbitration, order intents or
execution. The writer receives copies of values after the decision stages have
already run. Its return value is telemetry only.

## Integration points in the observed V5 flow

1. **Open receipt** after `run_id`, runtime paths and `DecisionAudit` are
   created in `main.py`. Capture code/config/image identity and a stable
   `receipt_id`.
2. **Market snapshot** after `_validate_market_data_snapshot` succeeds. Record
   the last completed bar timestamp per symbol, the conservative
   `market_data_cutoff`, universe, and a content-addressed market-input index.
   Do not copy all bars into the receipt JSON.
3. **Model and portfolio state** immediately after `pipe.run(...)`. Capture
   Alpha factor snapshots and weights, dynamic-IC state, regime/HMM/RSS/funding
   state, positions/cash, optimizer inputs and raw/post-risk targets. Existing
   `DecisionAudit` and `PipelineOutput` fields are source evidence.
4. **Final intents** after order arbitration, live preflight, Quant-Lab guard
   and order-lifecycle annotation, but immediately before
   `exec_engine.execute(orders)`. This ensures the recorded intents are the
   intents actually presented to execution.
5. **Finalize** after execution returns, or from a top-level `finally`/exception
   path. A failed decision must have `receipt_status=FAILED` plus the last
   completed stage and a redacted error summary. Execution results may be
   finalized by a later append-only event if fills are asynchronous.

The proposed integration is additive. It must be reviewed in V5-prod as a
separate patch before any use; Audit v2.1 does not modify that repository.

## Atomic append-only storage

One receipt uses an immutable generation directory:

```text
reports/decision_receipts/YYYY/MM/DD/<receipt_id>.partial/
  manifest.json
  content/<sha256>.json.zst
  content/<sha256>.parquet
  events/<sequence>-<event_sha256>.json
```

The writer creates files with exclusive creation, writes and fsyncs content,
validates the schema and hashes, fsyncs the directory, and atomically renames
`<receipt_id>.partial` to `<receipt_id>`. Existing final directories are never
overwritten. Duplicate calls with the same receipt/content hashes are no-ops;
different content under an existing ID is a high-severity integrity error.

Execution follow-ups are separate immutable events. A small append-only index
may point to receipts, but a broken index cannot invalidate a receipt.

## Non-blocking failure policy

- Receipt failures increment observable counters and emit a structured warning.
- Writer failure must never prevent stop-loss, risk-off, sell-only, position
  cleanup or exchange reconciliation.
- A bounded in-memory handoff and local spool are allowed; no unbounded queue.
- Opening/increasing exposure must not rely on receipt success either: the
  receipt is evidence, not a permission gate.
- Health telemetry reports `last_success_ts`, `last_error_ts`, `error_type`,
  `pending_spool_count`, `oldest_pending_age_seconds` and `write_latency_ms`.
- Secrets or schema violations reject the receipt payload, not the trade
  decision. The event must say that evidence capture degraded.

## Sensitive-data boundary

The serializer recursively rejects keys matching (case-insensitive)
`api_key`, `secret`, `passphrase`, `private_key`, `password`, `database_url`,
`dsn`, `authorization`, `cookie`, or `token`. It never serializes arbitrary
process environments or config objects. Allowed environment evidence is only
an explicit name/presence/digest allow-list.

Credentials are never hashed into the receipt: even a digest of a low-entropy
password is unsafe. Non-secret configuration values use canonical JSON SHA256.
Order/client objects must be converted through explicit allow-listed data
transfer objects; `repr()` and generic `asdict()` on external clients are
forbidden.

## Replay contract

Each large object is either an inline immutable value or a content reference
containing relative path, SHA256, schema version, media type, byte size and row
count where applicable. Replay must verify every referenced hash, code commit,
config hash, container digest, data snapshot ID and market-data cutoff before
recomputing. A replay result reports field-by-field equality for factor values,
target weights, Alpha/risk gates and order intents. It never calls an exchange
or submits an order.

The receipt can explain the recorded target only when every required field is
present and all references verify. Missing HMM/RSS/funding state must be an
explicit `UNAVAILABLE` value with reason, not a silently absent field.

## Review checklist before a V5 implementation

- Validate schema and recursive secret rejection with adversarial fixtures.
- Prove byte-for-byte stable canonicalization and stable `receipt_id`.
- Test power loss before/after content fsync and final rename.
- Test duplicate writes, content collision, spool recovery and disk-full mode.
- Inject writer exceptions before exits and confirm sell/stop-loss continues.
- Confirm receipt paths cannot traverse the configured root and reject symlinks.
- Benchmark the writer with maximum universe/features; keep it off the latency
  path via bounded copy/spool.
- Verify no production environment or systemd unit enables the feature by
  default.

## Audit v2.1 conclusion

This design closes the identified replayability gap at the contract level only.
It does not make the current V5 fully replayable, does not change the current
`PARTIALLY_REPLAYABLE` finding, and does not alter `production_alpha=FROZEN` or
`deployment_readiness=FAIL`.
