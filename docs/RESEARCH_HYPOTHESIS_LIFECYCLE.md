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

`REJECTED_DATA_QUALITY` is a terminal decision for the signed trial, not proof
that the economic hypothesis is false. An operator-approved hypothesis remains
`APPROVED_FOR_RESEARCH` so that a later source snapshot can be evaluated under
the same locked recipe. Content-addressed task identity prevents duplicate work
for unchanged inputs. A compatibility repair reopens only rows that the legacy
publisher marked `REJECTED` when every completed trial decision was
`DATA_BLOCKED` or `REJECTED_DATA_QUALITY`; signal, leakage, multiplicity, and
overfit failures remain terminal.

Approval requires confirmed real data, an operator identity, UTC approval time,
bounded recipes/horizons, and complete falsification/stopping rules. AI outputs
remain separate `AI_RESEARCH_DRAFT` rows and cannot transition directly.

Changing economic mechanism, data, formula, direction, universe,
neutralization, cost, split, or portfolio semantics requires a new hypothesis or
trial identity. Published historical versions are never rewritten.
