# Feature Publishing

`quant-lab` publishes research features from lake data only. The v0.1 feature
publisher reads `silver/market_bar`, uses closed bars, and writes reusable
feature rows to `gold/feature_value`.

It does not generate trading signals, orders, or execution instructions.

## Core Feature Set

`feature_set=core`, `feature_version=v0.1` publishes:

- `close_return_1`
- `close_return_4`
- `close_return_24`
- `rolling_volatility_24`
- `rolling_volatility_72`
- `volume_zscore_24`
- `range_bps`
- `close_position_in_range`
- `dollar_volume`
- `liquidity_proxy`

All rolling and shifted features are grouped by `symbol` and `timeframe`, so
values never mix instruments or bar intervals.

## Anti-Leakage Rules

- Inputs must be closed `market_bar` rows.
- Feature timestamps are the source bar timestamps.
- Feature computation only uses the current bar and historical bars.
- One-bar decision delay is supported by `validate_feature_timestamps`.
- Null values are explicit and include `is_valid=false` or an `invalid_reason`.

## Lake Outputs

- `gold/feature_value`
- `gold/feature_coverage_daily`
- `gold/feature_anomaly_daily`

Every feature row records:

- `input_dataset_version`
- `input_hash`
- `code_version`
- `created_at`
- `source`

## CLI

```bash
qlab publish-features \
  --lake-root /var/lib/quant-lab/lake \
  --feature-set core \
  --feature-version v0.1 \
  --timeframe 1H \
  --symbols BTC-USDT,ETH-USDT,SOL-USDT,BNB-USDT

qlab feature-health \
  --lake-root /var/lib/quant-lab/lake \
  --feature-set core \
  --date 2026-05-11
```

## API

```bash
curl 'http://127.0.0.1:8027/v1/features/latest?feature_set=core&symbols=BTC-USDT&timeframe=1H'
```

The endpoint is read-only and returns the latest published feature values.
