# Alpha Factory Integration

Alpha Factory may consume factor inputs only when
`gold/factor_research_generation.json` is present, valid, not from the future,
fresh enough, and verified against every published dataset's generation metadata
and row count.

The Alpha task and result bind:

- Factor Research generation ID and digest;
- hypothesis-registry and trial-ledger digests;
- generation as-of date and publish time;
- the exact hypothesis IDs.

Only current `SIGNAL_VALID` and `PAPER_CANDIDATE` research rows are eligible for
the bridge. `PORTFOLIO_FAIL`, retired, data-blocked, historical-only, stale, or
unbound rows cannot enter Alpha validation. This binding does not authorize
Paper, canary, or live trading; Alpha promotion remains cloud-derived and
research-only.
