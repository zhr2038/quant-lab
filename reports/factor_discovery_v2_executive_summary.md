# Factor Discovery v2 Executive Summary

## Decision

The refactor replaces automatic single-feature enumeration with a bounded,
hypothesis-driven research system. It is a research-quality correction, not a
claim that a profitable factor has been found.

## What changed

- Every executable calculation is bound to a versioned `hypothesis_id` and an
  immutable `trial_id`.
- The default agenda contains four independent hypotheses. Only two have
  confirmed data and approval to run; funding and microstructure remain
  `DATA_BLOCKED`.
- Each approved hypothesis has at most three recipes and three horizons. The
  global ledger budget is 54 trials.
- Every submitted, failed, invalidated, rejected, and completed trial remains in
  the append-only ledger and counts toward its multiple-testing family.
- Overlapping forward labels use HAC, non-overlapping estimates, block bootstrap,
  permutation evidence, Holm correction, and BH FDR. The old naive t-stat no
  longer has decision authority.
- Signal validity, long-only spot portfolio validity, and deployment readiness
  are separate states.
- Beta, liquidity, momentum, long-run volatility, regime, and symbol fixed
  effects are tested before an edge can be called incremental.
- PBO and Deflated Sharpe diagnostics block Paper review when overfit evidence is
  bad or inconclusive.
- Heavy computation runs as `task_type=factor_research` on the existing signed
  NAS Research Plane. No new queue or worker was created.
- Cloud validates the signed result and atomically publishes one authoritative
  Factor Research generation. Alpha Factory must bind to that exact generation.
- AI Stage 2 emits human-reviewable hypothesis, data-collection, attribution,
  and code-review drafts only. It cannot register, execute, or promote a trial.

## Safety

All new records carry `research_only=true`, `live_order_effect=none`,
`automatic_promotion=false`, and `max_live_notional_usdt=0`. V5 was inspected
read-only and was not modified. Paper ACK/Tracker, risk permission, canary,
enforce, capital, positions, and exchange order paths are outside this change.

## Acceptance result

Two final-commit NAS shadow tasks passed signature, snapshot, Anti-Leakage,
strict result validation, atomic publication, resource, exact generation
binding, and zero-live-effect checks. The weekly Factor request timer was
enabled only after that acceptance.

An additional operational commit bounded the frequent Web-derived refresh to
40 declared inputs instead of all 190 registered datasets. Production runtime
fell from a 45-minute timeout to 45.56 seconds in isolation, with no swap. The
complete V5 research refresh then finished successfully in 704.4 seconds. This
fix changes refresh resource behavior only and is not new Factor evidence.

The current research verdict is intentionally negative: all eight current
trials are `REJECTED_DATA_QUALITY`, with zero `SIGNAL_VALID` and zero
`PAPER_CANDIDATE`. This is successful platform acceptance, not successful alpha
discovery. The review branch is deployed for research shadow only; GitHub main
is unchanged and no live-readiness claim is made.
