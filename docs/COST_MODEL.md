# Cost Model

`quant-lab` calibrates read-only cost datasets from OKX direct data. The model
is research infrastructure only; it does not place, cancel, amend, transfer, or
withdraw.

## Inputs

Primary inputs:

- `silver/fill_event`: normalized OKX read-only private fills.
- `silver/account_bill`: normalized OKX read-only private bills.
- `silver/order_event`: normalized OKX read-only private order history.
- `silver/orderbook_snapshot`: OKX public WebSocket order book snapshots.
- `silver/trade_print` and `silver/market_bar`: public market context.

## Outputs

- `gold/cost_bucket_daily`
- `gold/cost_health_daily`

## Cost Source Priority

`source` is intentionally explicit:

- `actual_okx_fills_and_bills`: fill data has usable fee evidence and
  `sample_count >= min_sample_count`.
- `actual_okx_fills_fee_missing`: fill data exists, but fee/bill evidence is
  incomplete, or the sample is too small for a trusted actual cost bucket.
- `public_spread_proxy`: no fills are available; cost uses public bid/ask spread
  only.
- `global_default`: no usable fill or spread data exists.

`public_spread_proxy` is not actual all-in trading cost. It is a market
liquidity proxy.

## Fallback Flags

`fallback_level` may include:

- `FEE_MISSING`
- `BILLS_MISSING`
- `SAMPLE_TOO_SMALL`
- `SLIPPAGE_UNKNOWN`
- `SPREAD_PROXY`
- `SPREAD_MISSING`
- `PUBLIC_SPREAD_PROXY`
- `GLOBAL_DEFAULT`

When no order/reference price is available, slippage is not faked:

- `slippage_bps_p50/p75/p90 = 0`
- `fallback_level` includes `SLIPPAGE_UNKNOWN`

## Buckets

Notional buckets:

- `0-1k`
- `1k-10k`
- `10k-100k`
- `100k+`
- `all`

The `all` bucket gives strategy consumers a stable fallback for a symbol when a
specific bucket is sparse.

## Cost Health

`gold/cost_health_daily` records:

- actual rows
- proxy rows
- global default rows
- fallback ratio
- symbols with actual costs
- symbols with proxy-only costs
- symbols missing costs

Status rules:

- empty cost table -> `CRITICAL`
- all `global_default` -> `CRITICAL`
- all `public_spread_proxy` -> `WARNING`
- fallback ratio above `0.5` -> at least `WARNING`
- fallback ratio above `0.8`, unless all rows are proxy-only, -> `CRITICAL`
- actual rows with sufficient samples -> `OK` or `WARNING`

## Commands

```bash
qlab calibrate-costs \
  --lake-root /var/lib/quant-lab/lake \
  --day 2026-05-11 \
  --min-sample-count 30

qlab cost-health \
  --lake-root /var/lib/quant-lab/lake \
  --day 2026-05-11
```

The read-only API endpoint `GET /v1/costs/estimate` reads
`gold/cost_bucket_daily`. If the dataset is empty, it returns
`source=global_default` and `fallback_level=GLOBAL_DEFAULT`.
