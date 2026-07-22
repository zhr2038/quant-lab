# Factor Research v2

Factor Research v2 is a bounded research platform:

```text
Cloud hypothesis registry + immutable trial ledger
  -> signed immutable snapshot and task
  -> existing NAS Research Worker pure compute
  -> signed result + all-PASS anti-leakage
  -> strict cloud validation
  -> atomic Gold generation
  -> generation-bound Alpha Factory
```

It replaces automatic feature enumeration with explicit economic hypotheses.
The platform separates `signal_validity`, `portfolio_validity`, and
`deployment_readiness`. It can reject a cross-sectional signal as unsuitable for
the V5 spot long-only portfolio without erasing the signal evidence.

The authoritative datasets are:

- `gold/research_hypothesis_registry`
- `gold/research_trial_ledger`
- `gold/factor_definition`, `factor_value`, `factor_evidence`, `factor_candidate`
- `gold/factor_attribution`
- `gold/factor_portfolio_validation`
- `gold/factor_retirement`
- `gold/factor_external_audit_evidence`
- `gold/factor_research_generation.json`

Everything is research-only and has zero live-order effect.
