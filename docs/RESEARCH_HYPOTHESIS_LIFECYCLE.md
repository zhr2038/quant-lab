# Research Hypothesis Lifecycle

The lifecycle is versioned and fail-closed:

```text
DRAFT
  -> APPROVED_FOR_RESEARCH
  -> RUNNING
  -> SIGNAL_VALID
  -> PORTFOLIO_FAIL | PAPER_CANDIDATE

Any state may end at DATA_BLOCKED, REJECTED, RETIRED, or SUPERSEDED.
```

Approval requires confirmed real data, an operator identity, UTC approval time,
bounded recipes/horizons, and complete falsification/stopping rules. AI outputs
remain separate `AI_RESEARCH_DRAFT` rows and cannot transition directly.

Changing economic mechanism, data, formula, direction, universe,
neutralization, cost, split, or portfolio semantics requires a new hypothesis or
trial identity. Published historical versions are never rewritten.
