# Factor Attribution Design

Each trial reports raw Rank IC and residual Rank IC after the available controls:

- symbol fixed effects;
- market beta;
- liquidity;
- momentum;
- long-run volatility;
- size proxy and regime score when available.

Missing controls are explicit warnings. The system also records the dominant
symbol and maximum absolute contribution share. A result driven by one symbol or
substantially explained by controls cannot be called an incremental edge.

`low_vol_20d` is split into structural cross-sectional low volatility and
dynamic within-symbol volatility change. The historical audit remains visible,
but only the new decomposed trials may produce current evidence.
