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
