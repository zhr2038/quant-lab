# Current Factor Discovery Root Cause

The pre-change production system was an enumeration and ranking engine rather
than a controlled research program. Production evidence captured on 2026-07-19
showed 22 factor definitions, 10 automatically generated `auto.single.*`
definitions, 88 factor/horizon tests on the latest day, and seven latest
`PAPER_READY` candidates under a weak, overlap-naive threshold.

The root causes were:

1. Unbounded feature-to-factor registration without an economic hypothesis.
2. Post-hoc best-horizon selection and no immutable trial denominator.
3. Adjacent overlapping labels treated as independent observations.
4. No Holm/BH correction, blind split, permutation test, PBO, or DSR.
5. No beta, liquidity, symbol fixed-effect, or residual attribution.
6. Synthetic long-short signal evidence conflated with V5 spot long-only
   portfolio validity.
7. Latest cost rows attached to history without point-in-time provenance.
8. Cloud-local heavy computation and Alpha Factory consumption of unversioned
   latest outputs.
9. AI and Web surfaces rewarded proposal and candidate volume.

The full baseline, counts, code paths, and safety boundary are preserved in
`docs/current_factor_discovery_root_cause.md` and
`artifacts/current_factor_discovery_root_cause.json`.

## Shadow-deployment finding

The first production Shadow request exposed a separate point-in-time universe
defect. `gold/expanded_universe_quality` is an operational current snapshot: on
2026-07-19 it contained 94 rows for that day only. The 72-hour label embargo
made the research end date 2026-07-16, so the snapshot correctly found no
historical quality row. Reusing the 2026-07-19 membership for older decisions
would introduce future selection bias.

Factor Research therefore uses a versioned `spot-prior-closed-bars-v1`
universe. Membership for UTC day D is computed on NAS only from at least 18
positive-price closed 1H bars and positive quote volume observed during UTC day
D-1. Membership is valid for D only and is never carried forward when a day is
missing. The operational quality snapshot remains available to its existing
Web and Alpha Factory consumers, but it is not authoritative historical Factor
Research evidence.

Production `silver/market_bar` currently spans about 95 days, not the requested
730 days. This does not block a true infrastructure Shadow run, but it must
produce insufficient-data or non-promoting evidence until audited history is
backfilled. No result from this window may be described as a two-year finding.
