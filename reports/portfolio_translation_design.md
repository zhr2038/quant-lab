# Portfolio Translation Design

Factor signal evidence is not deployability evidence. The portfolio layer uses
the pre-registered V5-compatible spot rule:

- long-only Top-N or absolute timing;
- equal or score weights;
- fixed holding period;
- minimum order and tradability constraints;
- cash residual when a target cannot be bought;
- point-in-time fees and slippage;
- BTC and dynamic-universe benchmarks;
- concentration HHI, beta, Sortino, Calmar, and max drawdown.

Portfolio validity fails when after-cost return is not positive, edge/cost is at
most 1.5, validation or blind return is missing/negative, either benchmark wins,
drawdown exceeds the registered limit, or one symbol contributes more than 50%.
A statistically valid cross-sectional signal may therefore correctly end as
`PORTFOLIO_FAIL`.
