# Codex Tasks

Each PR must preserve the hard boundary: `quant-lab` is OKX-first and
read-only. It does not trade, does not use trading-capable OKX credentials, and
does not replace V5/V7 execution, reconcile, or kill-switch logic.
Strategy-facing APIs remain GET-only.

## PR 1 Repository Initialization

Objective:

Initialize the Python project, contracts, CLI, read-only FastAPI scaffold,
tests, linting, and CI.

Implementation notes:

- Create `pyproject.toml`, package layout, `README.md`, docs, and CI.
- Add Pydantic v2 models for market bars, costs, alpha evidence, gate
  decisions, and risk permissions.
- Add deterministic example CLI and API endpoints.
- Assert that `/v1` API routes expose no POST, PUT, PATCH, or DELETE methods.

Acceptance commands:

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check .
qlab gate-example
```

## PR 2 OKX Public Market Data Connector

Objective:

Make OKX public REST the default market-data source for `market_bar` lake
ingest.

Implementation notes:

- Implement OKX public GET endpoints for instruments, candles, history candles,
  ticker, and order book.
- Do not require an API key for public data.
- Normalize closed OKX candles into canonical `market_bar` records.
- Enforce closed-bar filtering and UTC timestamps.
- Publish to `lake/silver/market_bar`.
- Do not add private REST, private WebSocket, order, transfer, withdrawal, or
  account mutation behavior.

Acceptance commands:

```bash
pytest -q tests/test_okx_public.py
ruff check .
qlab okx-fetch-candles --inst-id BTC-USDT --bar 1H --market-type SPOT --lake-root /tmp/quant-lab-demo-lake
```

## PR 3 Parquet Publisher

Objective:

Publish validated bronze, silver, and gold datasets as deterministic Parquet
lake files.

Implementation notes:

- Add write helpers for partitioned Parquet outputs.
- Preserve stable logical keys for idempotent repeated publishes.
- Include dataset metadata such as schema version, generated time, source, and
  partition keys.
- Keep writes limited to ingest/research jobs. API and strategy clients read
  only.

Acceptance commands:

```bash
pytest -q tests/test_lake_publish.py
ruff check .
```

## PR 4 OKX Read-Only Private Cost Backfill

Objective:

Optionally ingest real OKX fills, bills, or order history using read-only
private GET endpoints for cost-model calibration.

Implementation notes:

- Require explicit configuration before enabling this path.
- Validate operator documentation states the OKX key must have Read permission
  only.
- Reject or refuse configurations intended for Trade or Withdraw permissions.
- Never log or write API keys, secret keys, or passphrases to lake files.
- Implement only read-only GET endpoints needed for fills, bills, or order
  history.
- Do not implement order placement, cancellation, amendment, funds transfer,
  withdrawal, or account mutation.

Acceptance commands:

```bash
pytest -q tests/test_okx_readonly_private.py
ruff check .
```

## PR 5 Feature Registry

Objective:

Create a feature registry and latest-feature publication path over OKX-first
silver datasets.

Implementation notes:

- Define feature metadata: name, version, owner, inputs, lookback, and output
  type.
- Ensure every feature declares whether it is closed-bar safe.
- Publish `feature_value` rows to gold with `observed_at` and `available_at`.
- Enforce one-bar decision delay for research/backtest consumers.

Acceptance commands:

```bash
pytest -q tests/test_features.py
ruff check .
```

## PR 6 Gate Engine Expansion

Objective:

Expand gate evaluation beyond default thresholds while preserving deterministic
and auditable decisions.

Implementation notes:

- Support configurable thresholds from `/etc/quant-lab/config.yaml`.
- Store gate inputs, outputs, reasons, source dataset versions, and threshold
  versions.
- Keep statuses limited to `DEAD`, `QUARANTINE`, `PAPER_READY`, and
  `LIVE_READY`.
- Add tests for look-ahead prevention and edge/cost failure paths.

Acceptance commands:

```bash
pytest -q tests/test_gates.py
ruff check .
```

## PR 7 V5/V7 qlab-client Integration

Objective:

Add V5/V7-side clients that read `quant-lab` decisions without giving
`quant-lab` execution authority.

Implementation notes:

- Clients use GET-only endpoints.
- Clients handle unavailable `quant-lab` responses with explicit, logged
  fallback behavior.
- V5/V7 remain responsible for order placement, cancellation, reconcile, and
  kill-switch actions.
- Do not give `quant-lab` trading-capable exchange credentials.

Acceptance commands:

```bash
pytest -q tests/test_api_read_only.py
pytest -q tests/test_client.py
ruff check .
```

## Optional Legacy PR V5 Reports Importer

Objective:

Keep the V5 reports adapter available as optional legacy importer tooling.

Implementation notes:

- Inspect old V5 report directories read-only.
- Normalize legacy report metadata when needed for comparison studies.
- Do not treat V5 reports as the primary data source.
- Do not mutate, delete, repair, or rewrite V5 files.

Acceptance commands:

```bash
pytest -q tests/test_v5_ingest.py
ruff check .
qlab inspect-v5 tests/fixtures/v5_reports
```
