# AGENTS.md

These instructions apply to the entire `quant-lab` repository.

## Project Identity

`quant-lab` is an OKX-first, read-only quantitative research platform running
on `qyun2.hrhome.top`. It is a research middle layer, not a trading bot.

Production access:

- Host: `qyun2.hrhome.top`
- SSH user: `ubuntu`
- Do not commit SSH passwords, OKX credentials, API secrets, passphrases, or
  private keys.

## OKX-First Source Design

The primary data chain is:

```text
OKX public REST
OKX public WebSocket
optional OKX read-only private GET
  -> lake
  -> feature/cost/backtest/evidence/gate
  -> read-only API
  -> V5/V7/future strategies
```

OKX public data does not require an API key. Optional OKX private credentials
may be configured only for read-only private GET endpoints used to backfill real
fills, bills, or order history for cost models and evidence.

V5 reports are optional legacy import artifacts. They are not the main data
source.

## Hard Boundaries

1. `quant-lab` MUST NOT place orders.
2. `quant-lab` MUST NOT cancel orders.
3. `quant-lab` MUST NOT amend, repair, or mutate live orders.
4. `quant-lab` MUST NOT transfer funds or withdraw assets.
5. `quant-lab` MUST NOT modify account configuration.
6. `quant-lab` MUST NOT maintain live exchange positions as source of truth.
7. `quant-lab` MUST NOT replace V5/V7 execution, reconcile, or kill-switch logic.
8. Strategy-facing APIs must be read-only. Use GET endpoints for external strategy consumption.
9. V5, V7, and future strategies read from `quant-lab`; they do not write gold research tables.
10. Optional OKX private keys must have Read permission only.
11. Optional OKX private keys MUST NOT have Trade permission.
12. Optional OKX private keys MUST NOT have Withdraw permission.
13. Optional OKX credentials MUST NOT be committed, logged, or written to lake files.

## Core Goal

Build a reusable OKX-first research hub that unifies market data, features,
cost models, statistical gates, research evidence, and risk permissions.

## Technology Constraints

- Use Python 3.11+.
- Use FastAPI for the read-only HTTP API.
- Use Pydantic v2 for contracts.
- Use DuckDB for local query/catalog access.
- Use Polars for vectorized and lazy data processing.
- Use Typer for CLI.
- Use Pytest for tests.
- Use Ruff for linting.
- Prefer Parquet lake files for storage.
- Avoid Redis, Celery, Kubernetes, Airflow, Kafka, Delta Lake, and Postgres in
  v0.1 unless explicitly requested.

## Architecture

- `bronze`: raw OKX public snapshots, optional OKX read-only private snapshots,
  and optional legacy importer snapshots.
- `silver`: normalized validated datasets such as `market_bar` and cleaned cost observations.
- `gold`: features, costs, evidence, gate decisions, and conclusions.
- APIs and clients must read from lake/catalog only.
- Ingest and research jobs may write lake files.
- Strategy consumers must be read-only.

## Required Concepts

- `market_bar`
- `feature_value`
- `cost_bucket_daily`
- `alpha_evidence`
- `gate_decision`
- `risk_permission`

## Gate Statuses

- `DEAD`
- `QUARANTINE`
- `PAPER_READY`
- `LIVE_READY`

## Risk Permissions

- `ALLOW`
- `SELL_ONLY`
- `ABORT`

## Quality Rules

- No look-ahead leakage.
- Use closed-bar signals only.
- Apply a one-bar decision delay for backtest and research.
- Use UTC timestamps.
- Keep ingest idempotent.
- Keep tests deterministic.
- No test should require live network access.
- Any fallback behavior must be explicit and logged/auditable.

## Development Behavior

- Before coding, inspect the repository structure.
- Preserve existing project conventions where present.
- Make small coherent changes.
- Add or update tests for every behavior change.
- Run `pytest` and `ruff` before finishing when the project has runnable tests/lint configuration.
- Report exact commands run and their results.
