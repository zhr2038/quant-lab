# Factor Research v2 point-in-time cost Shadow

Status: **PASS for research-cost coverage; BLOCKED for deployment-cost coverage**

This was a read-only local replay against qyun2 source data. It did not publish a generation,
write the production Lake, dispatch a task, change V5, or create an order.

## Locked window

- Portfolio cost window: `2026-06-16T00:00:00Z` to `2026-07-16T00:00:00Z` (exclusive)
- Symbols: `BTC-USDT`, `ETH-USDT`, `SOL-USDT`, `BNB-USDT`
- Hourly decision samples: 2,880
- Qualified symbol/day spread aggregates: 128 of 128
- Minimum observations per qualified symbol/day: 1,229
- Median observations per qualified symbol/day: 1,434.5
- Research proxy round-trip floor: 30 bps
- Portfolio cost coverage threshold: 80%
- Trusted deployment cost coverage threshold: 80%

## Results

| Metric | Published cost table only | With causal 1m spread reconstruction |
|---|---:|---:|
| Point-in-time research cost coverage | 65.14% | 100.00% |
| Trusted actual/mixed coverage | 12.50% | 12.50% |
| Public proxy coverage | 52.64% | 87.50% |
| Reconstructed proxy coverage | 0.00% | 36.67% |
| Bootstrap coverage | 0.00% | 0.00% |
| Stale rate | 34.86% | 0.00% |

All four symbols reached 100% point-in-time research coverage. Trusted coverage was 0% for BTC
and BNB, 23.33% for ETH, and 26.67% for SOL.

## Verdict

The 30-day portfolio layer can now run without zero-cost substitution or hindsight-filled daily
cost rows. The signal layer remains governed by its independent historical-data gate. Deployment
readiness remains blocked because 12.5% trusted coverage is far below 80%; reconstructed public
spread evidence is deliberately ineligible for live-cost coverage.
