# Architecture

`quant-lab` is an OKX-first, read-only quantitative research middle layer. It
organizes OKX market data, optional OKX read-only private cost evidence,
feature generation, cost calibration, statistical gates, and risk permissions
so V5, V7, and future strategies can consume research outputs without
delegating execution to `quant-lab`.

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

OKX public data is the default source for market bars, ticker snapshots, order
book snapshots, and future public market-data streams. OKX public data does not
require an API key.

Optional OKX read-only private GET endpoints may be used only for real fills,
bills, or order-history backfill that improves cost models and research
evidence. Any private key used for this path must have Read permission only. It
must not have Trade or Withdraw permission.

## Lake Design

The storage model is a local Parquet lake with three zones:

- `bronze`: raw OKX public snapshots, raw OKX public WebSocket messages,
  optional OKX read-only private snapshots, and optional legacy importer
  snapshots.
- `silver`: normalized and validated datasets. Examples: canonical
  `market_bar`, cleaned cost observations, normalized ticker snapshots, and
  normalized order-book snapshots.
- `gold`: strategy-facing research outputs. Examples: `feature_value`,
  `cost_bucket_daily`, `alpha_evidence`, `gate_decision`, and
  `risk_permission`.

APIs and clients read from the lake or a DuckDB catalog over the lake. Ingest
and research jobs may write lake files. V5/V7 strategy consumers must not write
gold tables.

## Ingest Layer

The ingest layer is OKX-first:

- OKX public REST pulls closed candles, instruments, tickers, and public order
  books.
- OKX public WebSocket is the intended live public market-data stream path.
- Optional OKX read-only private GET imports fills, bills, or order history for
  cost evidence.
- The V5 reports adapter is optional legacy importer tooling. It may inspect
  old report directories without mutating them, but it is not the main data
  source.

Ingest jobs must be idempotent. Re-running an ingest over the same source input
should produce the same lake output or a clearly versioned replacement.

## Research Layer

The research layer creates gold outputs from silver datasets. It owns feature
generation, calibrated cost buckets, backtest and walk-forward evidence, gate
decisions, and risk permission proposals.

Research rules:

- Use UTC timestamps.
- Use closed-bar signals only.
- Apply a one-bar decision delay in backtests and research evaluation.
- Avoid look-ahead leakage.
- Make fallback behavior explicit, logged, and auditable.

## Read API Layer

The FastAPI service exposes read-only GET endpoints. It should read from lake
files or catalog tables only. Strategy-facing routes must not accept POST, PUT,
PATCH, or DELETE methods.

The API may publish datasets, latest features, cost estimates, alpha evidence,
gate decisions, and live risk permissions. It must not place orders, cancel
orders, amend orders, transfer funds, withdraw assets, repair live orders, or
mutate live execution state.

## V5/V7 Integration

V5 and V7 are strategy consumers of `quant-lab`; they are not the primary data
source. They continue to own real trading execution, order state,
reconciliation, and kill-switch logic.

Read-only integration flow:

1. `quant-lab` ingests OKX public data and optional OKX read-only private cost evidence.
2. `quant-lab` writes normalized silver/gold lake datasets.
3. V5/V7 clients read GET endpoints or generated read-only files.
4. V5/V7 decide how to act using their own execution controls.

## Permission Boundary

`quant-lab` must not contain trading-capable credentials. Optional OKX private
credentials are allowed only for read-only private GET data collection and must
be stored outside Git with strict filesystem permissions.

Forbidden:

- Trade permission on any OKX key used by `quant-lab`.
- Withdraw permission on any OKX key used by `quant-lab`.
- Order placement, cancellation, amendment, or repair.
- Funds transfer or withdrawal.
- Account configuration mutation.
- Treating live exchange positions as `quant-lab` source of truth.

Position truth, order state, reconciliation, and emergency shutdown remain in
the trading bots and their execution infrastructure.

## qyun2 Filesystem Permissions

Suggested service account: `quantlab`. Suggested deploy account: `ubuntu`.

Recommended layout:

```text
/opt/quant-lab
/etc/quant-lab/config.yaml
/var/lib/quant-lab/lake
/var/log/quant-lab
```

Recommended permissions:

- `/opt/quant-lab`: `ubuntu:quantlab`, mode `0750`; writable by deploy user,
  readable/executable by the service group.
- `/etc/quant-lab`: `root:quantlab`, mode `0750`; config files mode `0640`.
- `/etc/quant-lab/config.yaml`: may contain optional OKX read-only private
  credential references, but no Trade or Withdraw credentials.
- `/var/lib/quant-lab/lake`: `quantlab:quantlab`, mode `0750`; writable only by
  approved ingest/research jobs.
- `/var/lib/quant-lab/lake/gold`: no write permission for V5/V7 strategy users.
- `/var/log/quant-lab`: `quantlab:adm`, directory mode `0750`, log files `0640`;
  logs must not contain API secrets, secret keys, or passphrases.

Prefer API access over direct filesystem reads for strategy consumers.
