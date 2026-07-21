# Factor Factory

`quant-lab` Factor Factory turns reusable `gold/feature_value` rows into
versioned factors, tests them against delayed forward labels, and publishes a
daily review queue.

It is read-only research infrastructure. It does not generate live orders,
cancel orders, amend orders, write exchange state, or change V5/V7 execution.

Production heavy computation is dispatched through the signed NAS Research
Compute Plane described in [FACTOR_FACTORY_NAS_RESEARCH_PLANE.md](FACTOR_FACTORY_NAS_RESEARCH_PLANE.md).
The local CLI is fail-closed unless an operator explicitly sets
`QUANT_LAB_LOCAL_FACTOR_FACTORY_ENABLED=1` for a maintenance or parity run.

## Data Flow

```text
silver/market_bar
  -> gold/feature_value
  -> gold/factor_definition
  -> gold/factor_value
  -> gold/factor_evidence
  -> gold/factor_candidate
  -> gold/factor_correlation_daily
  -> expert pack / web readers
```

## Datasets

### `gold/factor_definition`

Versioned factor metadata:

- `factor_id`
- `factor_name`
- `factor_family`
- `factor_version`
- `feature_set`
- `feature_version`
- `input_features_json`
- `template`
- `params_json`
- `expression_json`
- `expression_hash`
- `status`
- `lookback_bars`
- `availability_lag_bars`
- `warmup_bars`
- `required_bars`
- `causal`
- `normalization`
- `owner`
- `enabled`

### `gold/factor_value`

Computed factor values:

- `ts` remains the historical event key for backward compatibility.
- `event_time` is the closed bar timestamp described by the factor value.
- `available_time` is when the factor value may be consumed by a strategy.
- `raw_value`
- `normalized_value`
- `rank_value`
- `value`
- `factor_status`
- `expression_hash`
- `data_version`
- `calculated_at`
- `is_valid`
- `invalid_reason`
- `quality_flags_json`

Values are computed only from existing `gold/feature_value` rows. Cross-sectional
normalization is grouped by `factor_id + factor_version + timeframe + ts`.

### `gold/factor_evidence`

Per-factor validation metrics:

- IC / Rank IC
- IC and Rank IC t-stat
- top and bottom quantile return
- long-only after-cost mean bps
- long-short after-cost mean bps
- turnover proxy
- max drawdown
- edge/cost ratio
- decision

Decision values:

- `KILL`
- `RESEARCH`
- `KEEP_SHADOW`
- `PAPER_READY`

`PAPER_READY` means paper-review candidate only. It is not live eligibility.

### `gold/factor_candidate`

Daily review queue. Every row has `manual_review_required=true`; the factory
never directly promotes a factor into live execution.

### `gold/factor_correlation_daily`

Pairwise factor correlation table for redundancy control. High correlation
should be manually reviewed before paper promotion.

## CLI

```bash
qlab publish-features \
  --lake-root /var/lib/quant-lab/lake \
  --feature-set core \
  --feature-version v0.1 \
  --timeframe 1H

qlab build-factor-factory \
  --lake-root /var/lib/quant-lab/lake \
  --date auto \
  --feature-set core \
  --feature-version v0.1 \
  --factor-version v0.1 \
  --timeframe 1H \
  --horizon-bars 4,8,24,72 \
  --decision-delay-bars 1 \
  --min-samples 100 \
  --top-quantile 0.2 \
  --cost-quantile p75 \
  --apply

qlab factor-factory-health \
  --lake-root /var/lib/quant-lab/lake
```

The production request path is `qlab request-factor-factory`; it seals a signed
full-history Snapshot and does not compute or publish Factor Gold locally.

## Anti-Leakage Rules

- Factors consume only previously published `feature_value`.
- `feature_value` itself is computed from closed `market_bar`.
- Every factor definition is fail-fast causal: `causal=false` is rejected.
- Every factor value separates `event_time` from `available_time`.
- Forward labels use `decision_delay_bars >= 1`.
- Validation joins factor timestamp to label `feature_ts`.
- `decision_ts` must be after `feature_ts`.
- `label_ts` must be after `decision_ts`.

## First Default Factors

- `core.close_return_24`
- `core.short_reversal_1`
- `core.volume_zscore_24`
- `core.range_bps`
- `core.close_position_in_range`
- `core.liquidity_proxy`
- `core.momentum_vol_adjusted_24`
- `core.volume_momentum_4`
- `core.range_vol_ratio`
- `core.range_close_location`
- `core.liquidity_adjusted_momentum_24`
- `core.mean_reversion_vol_adjusted_4`

The ordinary local registry defaults to the curated factors. The signed
`PARITY_FULL` NAS compatibility plan explicitly includes sorted
`auto.single.<feature_name>` factors to preserve the audited pre-migration
semantics; that choice is recorded in the plan and cannot be expanded by NAS.
