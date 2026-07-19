# Factor Retirement Policy

Retirement disables a factor without deleting evidence. A factor is retired or
blocked when independent audit fails, multiple-testing governance fails,
required real data is absent, a definition is superseded, or a new generation
rejects the signal.

`gold/factor_retirement` records factor, family, status, enabled flag, reason,
signal and portfolio validity, source audit, UTC time, and safety fields.

Retired and `DATA_PENDING` factors:

- remain visible in Web and expert exports;
- cannot seed new trials automatically;
- cannot enter current Alpha Factory inputs;
- cannot be promoted to Paper, canary, or live;
- may return only through a new approved hypothesis/version with new evidence.
