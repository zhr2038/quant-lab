# Hypothesis Registry Design

`gold/research_hypothesis_registry` is the cloud-owned control plane for Factor
Research v2. Its primary key is `(hypothesis_id, hypothesis_version)` and its
definition digest excludes lifecycle timestamps and approval state.

An executable hypothesis must define:

- economic mechanism, payer, persistence, and decay;
- data requirements and confirmed availability;
- one to three bounded feature recipes and horizons;
- explicit direction, universe, neutralization, benchmark, and cost model;
- falsification, stopping, and success conditions;
- known overlap and related factors;
- human approval with UTC timestamp.

Changing the definition requires a new version. A single hypothesis cannot have
multiple active versions. AI drafts cannot become executable in place; a human
must re-register them as a new operator-owned hypothesis.

Budgets are fail-closed: at most two active hypotheses per family, six active
hypotheses globally, three variants per hypothesis, and 54 trials globally.
