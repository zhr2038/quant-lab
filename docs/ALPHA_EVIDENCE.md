# Alpha Evidence

`quant-lab` generates alpha evidence from read-only lake datasets. It does not
place orders, cancel orders, amend orders, or produce execution instructions.

## Inputs

Required datasets:

- `silver/market_bar`: closed OKX market bars.
- `gold/feature_value`: reusable versioned research features.

Optional dataset:

- `gold/cost_bucket_daily`: cost estimates used to estimate after-cost research
  returns. If missing, research uses an explicit global default cost and records
  a warning.

## Research Spec

`AlphaResearchSpec` defines a reproducible research run:

- `alpha_id`, `version`
- `feature_set`, `feature_version`, `feature_names`
- `timeframe`
- `label_horizon_bars`
- `decision_delay_bars`, default `1`
- `universe_id`
- `strategy`, default `v5`
- `min_samples`, `min_coverage`
- `cost_quantile`, one of `p50`, `p75`, `p90`
- `top_quantile`

## Anti-Leakage Rules

Forward labels are built per `symbol` and `timeframe`.

For a feature at `feature_ts`:

- `decision_ts = feature_ts + decision_delay_bars`
- `label_ts = feature_ts + decision_delay_bars + label_horizon_bars`
- `forward_return = close(label_ts) / close(decision_ts) - 1`

The default one-bar decision delay means a decision at `10:00` can use features
from `09:00` or earlier, not the current decision bar.

## Metrics

Evidence includes:

- coverage
- IC and IC t-stat
- rank IC and rank IC t-stat
- simple long-only after-cost OOS Sharpe
- max drawdown
- turnover proxy
- edge/cost ratio
- profitable fold ratio
- train/OOS decay

The v0.1 OOS simulation ranks symbols by `alpha_score`, selects the top
quantile, and computes equal-weight forward returns after subtracting the
configured cost quantile. It is research accounting only.

## Gate Publishing

`qlab build-alpha-evidence` writes:

- `gold/alpha_evidence`
- `gold/gate_decision`

The gate engine remains conservative:

- weak coverage, IC, t-stat, or edge/cost -> `DEAD`
- weak OOS evidence -> `QUARANTINE`
- research passes but paper observation is missing -> `PAPER_READY`
- only sufficient research and paper evidence can become `LIVE_READY`

Generated alpha evidence defaults to:

- `paper_days = 0`
- `paper_slippage_coverage = 0`

Therefore a new research run cannot become `LIVE_READY` until paper evidence is
added.

## Risk Permission Publishing

`qlab publish-risk-permission` reads:

- `gold/gate_decision`
- `gold/cost_health_daily` or `gold/cost_bucket_daily`
- `silver/market_bar`
- optional V5 telemetry health/compliance tables

Then it writes `gold/risk_permission`.

Rules:

- `DEAD` gate -> `ABORT`
- `QUARANTINE` gate -> `SELL_ONLY`
- `PAPER_READY` gate -> `ALLOW` with `allowed_modes=["paper"]`
- cost missing, stale, or high fallback -> `SELL_ONLY`
- stale or missing market data -> `ABORT`
- V5 gate compliance violation -> `ABORT`

## Commands

```bash
qlab build-alpha-evidence \
  --lake-root /var/lib/quant-lab/lake \
  --alpha-id v5.core.momentum \
  --version v0.1 \
  --feature-set core \
  --feature-version v0.1 \
  --feature-names close_return_24 \
  --timeframe 1H \
  --label-horizon-bars 4 \
  --decision-delay-bars 1 \
  --universe-id okx-major-spot \
  --strategy v5 \
  --cost-quantile p75

qlab publish-gate-decisions --lake-root /var/lib/quant-lab/lake --strategy v5

qlab publish-risk-permission \
  --lake-root /var/lib/quant-lab/lake \
  --strategy v5 \
  --version 5.0.0

qlab research-health --lake-root /var/lib/quant-lab/lake
```

## API

Read-only endpoints:

- `GET /v1/research/alpha/{alpha_id}`
- `GET /v1/gates/decision/{alpha_id}`
- `GET /v1/risk/live-permission?strategy=v5&version=5.0.0`

If a gate decision is missing, the API returns a conservative `QUARANTINE`
placeholder instead of a demo live-ready decision.
