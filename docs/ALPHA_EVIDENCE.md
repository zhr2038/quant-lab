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

## Strategy Evidence And Alpha Discovery

Feature-level alpha evidence is not the only V5 research output. `quant-lab`
also builds `gold/strategy_evidence_sample` and `gold/strategy_evidence` from
read-only V5 telemetry, shadow outcomes, blocked candidates, router decisions,
probe diagnostics, and quant-lab cost usage.

The first strategy candidates are evaluated independently:

- `v5.btc_leadership_probe_strict`
- `v5.sol_protect_exception`
- `v5.alt_impulse_shadow`
- `v5.swing_f4_f5_alpha6`
- `v5.f3_dominant_entry`
- `v5.mean_reversion_sideways`

Each sample records candidate context such as `f1` through `f5`,
`alpha6_score`, `alpha6_side`, regime, protect level, expected/required edge,
block reason, and cost source. Forward labels are generated with a one-bar
decision delay for 4h, 8h, 12h, 24h, 48h, 72h, and 120h horizons.

Candidate decisions are:

- `KILL`
- `KEEP_SHADOW`
- `PAPER_READY`
- `LIVE_SMALL_READY`

`LIVE_SMALL_READY` is never emitted when `sample_count < 30`. SOL protect
exceptions and alt impulse shadow evidence remain shadow/protective candidates
unless future policy explicitly changes their promotion cap. The strict BTC
leadership probe is matched only from explicit strict probe telemetry; broad BTC
leadership blockers are not mixed into that candidate.

Daily expert exports include the strategy evidence in both raw research form
and review-board form:

- `research/strategy_evidence.csv`
- `research/strategy_evidence_samples.csv`
- `reports/alpha_discovery_board.csv`
- `reports/strategy_evidence_summary.md`
- `reports/candidate_kill_list.csv`
- `reports/candidate_shadow_watchlist.csv`
- `reports/candidate_paper_ready.csv`

## V5 Candidate Event And Label Chain

V5 candidate discovery also has a canonical event/label chain:

- `reports/candidate_snapshot.csv` in each V5 follow-up bundle.
- `silver/v5_candidate_event` in the quant-lab lake.
- `gold/v5_candidate_label` for forward labels.
- `gold/v5_candidate_quality_daily` for row, feature, label, and cost-source
  checks.
- `gold/v5_candidate_outcome_summary` for grouped hindsight by
  `block_reason`, `strategy_candidate`, `symbol`, and `horizon_hours`.
- `gold/alpha_discovery_board` for the daily candidate decision board grouped
  by `strategy_candidate`, `symbol`, `regime_state`, and `horizon_hours`.

`candidate_snapshot.csv` captures every run/symbol candidate state, including
f1-f5, alpha6, final score, expected/required edge, cost, final decision, block
reason, and `strategy_candidate`. Missing `candidate_id` values are generated
from `run_id + symbol + strategy_candidate`.

Forward labels use closed market bars and the same one-bar decision delay used
by research evidence. The horizons are 4h, 8h, 12h, 24h, 48h, 72h, and 120h.
Each horizon records `gross_bps`, `net_bps_after_cost`, `mfe_bps`, `mae_bps`,
`win`, and `label_status`.

The daily expert pack includes:

- `v5/v5_candidate_events.csv`
- `v5/v5_candidate_labels.csv`
- `v5/v5_candidate_quality.csv`
- `v5/v5_candidate_outcome_summary.csv`

`qlab build-alpha-discovery-board` turns the candidate labels into the daily
decision panel. It reports sample counts, complete labels, average/median/p25
net bps, win rate, MFE/MAE, cost source mix, day stability, paper days, and a
decision of `KILL`, `RESEARCH_ONLY`, `KEEP_SHADOW`, `PAPER_READY`, or
`LIVE_SMALL_READY`. `LIVE_SMALL_READY` requires at least 60 samples, at least
14 paper days, no `global_default` cost source, and cannot be assigned to
`v5.alt_impulse_shadow` or `v5.sol_protect_exception`.

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

qlab build-strategy-evidence \
  --lake-root /var/lib/quant-lab/lake \
  --date auto \
  --min-live-samples 30

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
