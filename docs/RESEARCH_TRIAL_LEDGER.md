# Research Trial Ledger

The Trial Ledger is the denominator for statistical governance. Every attempted
combination of hypothesis version, recipe, horizon, universe, neutralization,
cost, split, seed, code, and data snapshot has one deterministic `trial_id`.

Rows are append-only by identity. Runtime fields may progress monotonically from
submitted to running and a terminal state. Failures are never deleted and remain
in Holm/BH families. Confirmatory parameters are locked before the blind window;
post-hoc changes invalidate that blind evidence.

Historical audit findings without immutable identity are stored separately as
external audit evidence and cannot satisfy current significance or promotion
requirements.
