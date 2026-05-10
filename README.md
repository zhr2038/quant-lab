# quant-lab

`quant-lab` is an OKX-first, read-only quantitative research middle platform for qyun2.

It builds the research data chain directly from OKX public market data, with an
optional OKX read-only private GET path for real fill/bill/order-history cost
calibration. V5, V7, and future strategies consume `quant-lab` outputs; they
continue to own real trading execution, reconciliation, and kill-switch logic.

## What It Does

- Pull OKX public REST data directly into the local lake.
- Support OKX public WebSocket ingestion as the intended live market-data path.
- Optionally ingest OKX read-only private GET data for real cost backfill.
- Unify market data, cost observations, research artifacts, and evidence.
- Generate reusable feature datasets for strategy research.
- Calibrate cost models by symbol, regime, notional size, and observed costs.
- Run research, backtest, and walk-forward workflows with closed-bar signals.
- Evaluate statistical gates for alpha readiness.
- Publish read-only research decisions and risk permissions.

## What It Does Not Do

- No order placement.
- No order cancellation.
- No order amendment or live order repair.
- No live execution.
- No funds transfer or withdrawal logic.
- No account configuration mutation.
- No replacement of V5/V7 reconcile or kill-switch logic.

## OKX Credentials

OKX public data does not require an API key.

If real cost backfill is enabled, `quant-lab` may use an OKX private key only
for read-only private GET endpoints such as fills, bills, or order history. That
key must have Read permission only.

Forbidden:

- Trade permission.
- Withdraw permission.
- Any order, transfer, withdrawal, or account mutation endpoint.
- Committing API keys, secret keys, passphrases, or private credentials to the
  repository.
- Logging or writing secrets to the lake.

## Main Data Chain

```text
OKX public REST
OKX public WebSocket
optional OKX read-only private GET
  -> lake
  -> feature/cost/backtest/evidence/gate
  -> read-only API
  -> V5/V7/future strategies
```

The V5 reports importer is optional legacy tooling. It is not the primary data
source for v0.1+.

## Quickstart

```bash
python -m pip install -e '.[dev]'
pytest -q
qlab gate-example
qlab okx-fetch-candles --inst-id BTC-USDT --bar 1H --market-type SPOT --lake-root /tmp/quant-lab-demo-lake
uvicorn quant_lab.api.main:app --host 127.0.0.1 --port 8027
qlab-web --host 127.0.0.1 --port 8501 --lake-root /tmp/quant-lab-demo-lake
```

The API is intended for read-only local or internal access. Strategy-facing
routes must use GET and must not mutate live execution state.

## qyun2 Suggested Layout

```text
/opt/quant-lab
/etc/quant-lab/config.yaml
/var/lib/quant-lab/lake
/var/log/quant-lab
```

Recommended use:

- `/opt/quant-lab`: checked-out application code and virtual environment.
- `/etc/quant-lab/config.yaml`: runtime configuration. If optional OKX
  read-only private credentials are enabled, keep them out of Git and restrict
  the file to the service account.
- `/var/lib/quant-lab/lake`: Parquet lake files for bronze, silver, and gold data.
- `/var/log/quant-lab`: API, ingest, and research job logs without secrets.

## Development Commands

```bash
ruff check .
pytest -q
qlab okx-fetch-candles --inst-id BTC-USDT --bar 1H --market-type SPOT --lake-root /tmp/quant-lab-demo-lake
```

Tests must be deterministic and must not require live network access.
