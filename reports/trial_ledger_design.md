# Trial Ledger Design

`gold/research_trial_ledger` is append-only by `trial_id`. A trial identity binds
the hypothesis version, formula and recipe hashes, direction, lookback, horizon,
universe, neutralization, point-in-time cost model, portfolio rule, chronological
split, blind period, seed, full code commit, data snapshot, and NAS task.

All trials set `counts_toward_multiple_testing=true`. Failed, rejected,
cancelled, and invalidated rows remain queryable. Status updates may add runtime
timestamps and decisions, but any identity-digest mutation is rejected.

Confirmatory parameters are locked before the blind period is opened. A post-hoc
change invalidates the blind trial. The chronological split is 60% development,
20% validation, and 20% blind confirmation with an embargo equal to the maximum
horizon.

Historical audits that lack immutable code, snapshot, and split identity are
preserved in `gold/factor_external_audit_evidence`; they are deliberately not
fabricated into new ledger rows and cannot promote a factor.
