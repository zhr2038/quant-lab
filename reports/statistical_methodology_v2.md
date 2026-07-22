# Statistical Methodology v2

## Dependence-aware evidence

For a horizon of `h` bars, adjacent forward labels overlap. The runtime reports:

- the descriptive naive t-stat, with no decision authority;
- Newey-West/HAC t-stat using a horizon-aware lag;
- every-`h` non-overlapping estimate;
- moving-block bootstrap confidence interval;
- deterministic permutation empirical p-value;
- effective sample size and sign consistency across major periods.

## Multiple testing

The complete retained trial family, including failed and rejected attempts,
feeds Holm-Bonferroni and Benjamini-Hochberg adjustments. A confirmatory signal
requires blind Rank IC above 0.03 and either HAC t-stat at least 3.0 or Holm
adjusted p-value at most 0.05. Non-overlapping direction, bootstrap direction,
permutation p-value at most 0.05, two major periods with the same sign, and no
single-symbol dominance above 50% are also required.

## Overfit diagnostics

CSCV estimates Probability of Backtest Overfitting. Deflated Sharpe accounts for
selection. Paper review requires `PBO <= 0.20` and DSR probability at least 0.95.
If too few variants exist, the result is
`INCONCLUSIVE_OVERFIT_DIAGNOSTICS`, never an implicit pass.
