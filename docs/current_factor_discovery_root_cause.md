# Current Factor Discovery Root Cause

This is the pre-change baseline for Factor Discovery v2. It combines current `main` code inspection with qyun2 Lake evidence captured on 2026-07-19. It does not claim that code not yet deployed is operational.

## Executive finding

The current Factor Factory is an enumeration and ranking system, not a controlled research program. It automatically converts every available feature into a single-feature factor, evaluates every factor over four horizons every day, uses an overlap-naive t-statistic, selects the best horizon after observing results, and can label that selection `PAPER_READY` at `rank_ic_tstat >= 1.0`. There is no immutable hypothesis registry, complete trial ledger, blind confirmatory split, multiple-testing correction, factor attribution, or long-only portfolio validity gate.

This explains the observed state: research output volume is high while evidence quality is too weak to support Paper promotion.

## Observed production scale

- 10 feature names in `gold/feature_value`, 88,080 rows.
- 22 factor definitions; 10 are `auto.single.*`.
- 193,688 factor-value rows; 88,040 belong to `auto.single.*`.
- 3,784 factor-evidence rows over 43 dates and four horizons.
- 946 factor-candidate rows; 430 are `auto.single.*`.
- Latest day: 22 factors x 4 horizons = 88 tests.
- Latest evidence decisions: 21 `PAPER_READY`, 26 `KEEP_SHADOW`, 26 `RESEARCH`, 15 `KILL`.
- Latest candidate decisions after best-horizon selection: 7 `PAPER_READY`, 9 `KEEP_SHADOW`, 6 `RESEARCH`.
- Historical evidence contains 385 `PAPER_READY` rows; historical candidates contain 170.
- `gold/research_hypothesis_registry`, `gold/research_trial_ledger`, and `gold/factor_research_generation` do not exist.

## Root causes

### 1. Unbounded feature enumeration

`discover_factor_specs()` in `src/quant_lab/factors/registry.py` first loads the built-in registry and then creates `auto.single.<feature_name>` for every available feature until `max_factors`. There is no hypothesis identity, approval state, variant budget, horizon budget, or research-family budget.

### 2. Weak promotion threshold

`_factor_decision()` in `src/quant_lab/factors/factory.py` returns `PAPER_READY` when coverage is at least 80%, Rank IC is merely positive, the ordinary t-stat is at least 1.0, long-short mean is positive, and edge/cost is above 1.0. The threshold is not confirmatory significance and is not sufficient for a forward Paper candidate.

### 3. Overlapping labels treated as independent

`compute_rank_ic()` groups by each decision timestamp and applies `mean / sample_std * sqrt(N)` to adjacent IC observations. For 4/8/24/72-bar forward labels, adjacent observations share future returns. The reported t-stat therefore overstates independent evidence.

### 4. Post-hoc horizon selection

`_candidate_frame_from_evidence()` sorts all tested horizons by score and promotes the best row. A factor's `best_horizon_bars` is selected after observing all four horizons, but no multiple-testing penalty or pre-registered horizon exists.

### 5. No complete trial ledger

Daily evidence rows are upserted by date/factor/horizon. They do not preserve a formal trial identity, planned split, confirmatory/exploratory status, random seed, parameter family, failed/cancelled attempt, or blind-period validity. Historical test count cannot be reconstructed reliably from winner tables.

### 6. No multiple-testing correction

The system does not calculate Holm-Bonferroni, Benjamini-Hochberg FDR, family-wide empirical p-values, or local FDR. Failed and abandoned experiments are not included in a denominator because there is no ledger.

### 7. No research/development/validation/blind separation

Factor evidence is computed over the full joined dataset. There is no pre-registered chronological split, blind lock, or rule that invalidates a blind period after parameters are changed.

### 8. Cross-sectional signal is conflated with spot deployability

The primary gate uses a synthetic long-short spread even though V5 is spot long-only. A relative rank signal can pass while Top-N long-only after-cost performance loses money. The current status taxonomy has no explicit `signal_validity`, `portfolio_validity`, or `deployment_readiness` separation.

### 9. Portfolio simulation is incomplete

The Factor Factory's `_portfolio_stats()` uses a quantile mean and a turnover proxy. It does not pre-register Top 1/3/5, cash residual, minimum order constraints, untradeable assets, actual two-sided turnover costs, BTC and dynamic-universe benchmarks, concentration HHI, beta, Sortino, Calmar, or symbol contribution limits.

### 10. Historical cost leakage risk

`_latest_cost_frame()` selects the latest cost row per symbol and attaches it to all historical observations. This is acceptable only as an explicitly conservative current research assumption; it is not point-in-time historical cost evidence and must not support deployment readiness.

### 11. No attribution or residual signal test

The current path has semantic factor dedupe and correlation clustering, but it does not decompose symbol fixed effects, market beta, liquidity, long-run volatility, momentum, or regime. In particular, `low_vol` structural cross-sectional identity and dynamic within-symbol volatility change are not separated.

### 12. No overfit diagnostics

PBO/CSCV, Deflated Sharpe Ratio, selection degradation, block bootstrap, permutation nulls, and non-overlapping sample estimates are absent from the production Factor Factory.

### 13. Cloud still performs heavy factor research

`deploy/systemd/quant-lab-v5-research-refresh.service` runs `qlab build-factor-factory` on qyun2 under a 4 GiB memory limit and 60% CPU quota. Factor Research is not a signed NAS task type.

### 14. Alpha Factory consumes unversioned latest factor outputs

The Alpha Snapshot includes `gold/factor_candidate` and `gold/factor_value`, and the worker recomputes Factor Bridge from them. There is no binding to a validated `factor_research_generation`, hypothesis-registry digest, or trial-ledger digest.

### 15. AI Stage 2 generates implementation-shaped candidates

The current AI contract emits `FactorProposal` and `PaperStrategyDraft`, with up to 8 factor proposals and 6 Paper drafts. Although they remain read-only, this is too close to automatic template production. Factor Discovery v2 must emit human-reviewable hypothesis, data-collection, attribution-experiment, and code-review drafts only.

### 16. Dashboard rewards volume

Web V2 emphasizes candidate count, formula count, `PAPER_READY`, composite candidates, and bridge candidates. It does not lead with independent hypotheses, total retained trials, budget utilization, corrected significance, attribution, PBO/DSR, portfolio failures, or data blockers.

## False-positive conclusion

Current `PAPER_READY` is a false-positive-prone research label. The code and the observed 7-of-22 latest factor count prove that it cannot be treated as a reliable Paper-candidate gate. Existing rows must remain as history but be retired from the new authoritative lifecycle.

## Required architectural correction

1. Register and approve a small number of versioned hypotheses.
2. Expand each hypothesis only into its pre-registered bounded variants and horizons.
3. Record every attempted trial in an immutable ledger, including failures.
4. Run causal, overlap-aware, multiple-testing-adjusted, attribution and portfolio validation on the shared NAS Research Plane.
5. Publish a signed, cloud-validated Factor Research generation.
6. Allow Alpha Factory to consume only that generation and only `SIGNAL_VALID` or `PAPER_CANDIDATE` factors.
7. Keep AI outputs as unapproved drafts and keep all trading effects at zero.

## Safety boundary

The correction is research-only. It must not modify V5 entry/exit behavior, Paper ACK/Tracker, Risk Permission, canary, enforce, capital, or exchange state. A reduction to zero newly valid factors is an acceptable and truthful result.
