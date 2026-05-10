# Web Dashboard

`quant-lab-web` is a Streamlit read-only observability dashboard for the
OKX-first quant-lab lake. It is intended for daily operational review of data
quality, market state, costs, alpha gates, strategy consumer permissions, and
expert export packs.

It does not trade. It does not expose strategy mutation controls. It does not
modify exchange state, strategy state, lake files, credentials, balances, or
orders.

## Start

Install the package with dashboard dependencies:

```bash
python -m pip install -e '.[dev]'
```

Run locally:

```bash
qlab-web --host 127.0.0.1 --port 8501 --lake-root /var/lib/quant-lab/lake
```

Run on qyun2 for internal access:

```bash
qlab-web --host 0.0.0.0 --port 8501 --lake-root /var/lib/quant-lab/lake
```

The dashboard reads the lake directly. It should be run behind trusted internal
network controls.

## Pages

### Overview

Use this page first. Check:

- `quant-lab status`: `OK`, `WARNING`, or `CRITICAL`.
- V5/V7 permission: `ALLOW`, `SELL_ONLY`, `ABORT`, or `UNKNOWN`.
- OKX public REST / WebSocket status.
- Optional OKX read-only private status.
- Latest market bar timestamp.
- Missing bar ratio.
- Cost fallback ratio.
- Alpha gate counts.
- Latest expert pack path.

### Data Health

Use this page to check the canonical `market_bar` dataset:

- latest bar per symbol/timeframe
- missing bars
- duplicate bars
- unclosed bars
- schema violations
- stale or missing datasets

Manual intervention is required when duplicate bars, unclosed bars, schema
violations, or unexpectedly stale core datasets appear.

### OKX Collectors

Use this page to review collector freshness:

- REST success count
- WebSocket message count
- latest successful collection timestamp
- reconnect count if available
- rate-limit warning count if available
- collector lag label

If REST and WebSocket are both stale, check qyun2 network, OKX status, and the
collector process logs.

### Market Regime

Use this page to inspect market state derived from public data:

- volatility regime
- spread bps from order book snapshots when available
- trade activity
- abnormal symbols

Funding rate and open-interest summaries are planned follow-ups once those
public datasets are published.

### Cost Model

Use this page to inspect `gold/cost_bucket_daily`:

- fee bps p50/p75/p90
- slippage bps p50/p75/p90
- spread bps p50/p75/p90
- total cost bps p50/p75/p90
- fallback level
- sample count

Manual review is required when fallback usage is high or sample counts are too
low for a strategy's target notional.

### Alpha Gates

Use this page to inspect gate decisions:

- alpha id and version
- status
- coverage
- IC metrics
- OOS metrics
- edge/cost ratio
- paper days
- reasons
- next action

Weak alpha should remain `DEAD` or `QUARANTINE` until evidence improves.

### Strategy Consumers

Use this page to inspect V5/V7 read-side integration:

- latest risk permission rows
- cost model version
- gate version
- permission
- allowed modes
- fallback evidence from legacy decision audits when present

V5/V7 continue to own real execution, reconciliation, balances, and kill-switch
logic. quant-lab only publishes data, cost estimates, gate decisions, and risk
permission signals.

### V5 Telemetry

Use this page to inspect the remote V5 bundle ingest and analysis outputs:

- latest bundle timestamp and sha256
- run, decision audit, trade, and roundtrip counts
- open and residual position counts
- kill-switch, reconcile, ledger, and auto-risk state
- high/medium issue counts
- config values not consumed at runtime
- high-score blocked and matured opportunity counts
- router reason top table
- next actions from `gold/strategy_health_daily`

V5 telemetry is audit and behavior data, not the primary market data source.
Raw bundles are kept in restricted archive only; the dashboard reads the
redacted archive and lake-derived gold datasets.

### Expert Exports

Use this page to inspect expert analysis packs:

- latest `quant_lab_expert_pack_YYYY-MM-DD.zip`
- manifest summary
- data quality summary
- expert questions
- available pack paths and file sizes

The default export directory is the sibling `exports` directory next to the
lake root, for example `/var/lib/quant-lab/exports`.

## Daily Checklist

1. Open Overview and confirm status is not `CRITICAL`.
2. Confirm latest market bar timestamps are current for expected symbols.
3. Confirm missing bar ratio, duplicate bars, and unclosed bars are acceptable.
4. Confirm OKX REST and WebSocket collectors are fresh.
5. Review cost fallback ratio and low-sample cost buckets.
6. Review `DEAD`, `QUARANTINE`, `PAPER_READY`, and `LIVE_READY` gate counts.
7. Confirm V5/V7 risk permissions match operator expectations.
8. Review V5 Telemetry for bundle freshness, kill-switch/reconcile/ledger
   status, high issues, and gate compliance warnings.
9. Open the latest expert pack and review data quality warnings and questions.

## Manual Intervention

Intervene when:

- market data is stale or missing
- duplicate or unclosed bars appear
- schema violations are present
- cost buckets are missing or mostly fallback/proxy
- an expected alpha moves to `DEAD` or `QUARANTINE`
- V5/V7 permission is `ABORT` unexpectedly
- V5 telemetry reports stale bundles, kill-switch enabled, reconcile failure,
  ledger failure, or gate compliance violations
- the latest expert pack is missing
- any dashboard output appears to expose credentials or private auth material

## Security Boundary

The dashboard reads only lake files and derived summaries. It must not expose
buttons or endpoints that mutate strategy state or exchange state. Optional OKX
read-only private datasets may be summarized, but credentials and private auth
material must never be displayed.
