# Production Schedule

`quant-lab` production jobs are read-only research jobs. They must not place,
cancel, amend, transfer, withdraw, or mutate exchange state.

## Required Ordering

Daily and recurring jobs must keep this order:

1. `qlab sync-v5-telemetry`
2. `qlab publish-risk-permission`
3. `qlab export-daily`

The ordering matters because `risk_permission` must be evaluated against the
latest V5 telemetry. If `risk_permission.created_at` is older than the latest
V5 bundle timestamp, the expert export reports:

```text
risk_permission_stale_vs_v5_telemetry
```

## Strategy Version

Use one stable pair across V5 and quant-lab:

```text
strategy=v5
version=<V5 QuantLabConfig.strategy_version>
```

Recommended values are `v1` or `5.0.0`, but do not mix them in the same
production run. If V5 telemetry exports `strategy_version` and it does not
match `gold/risk_permission.version`, the expert export reports a version
mismatch warning.

## Systemd Timers

The repository includes these timer templates:

- `deploy/systemd/quant-lab-v5-telemetry-sync.timer`
- `deploy/systemd/quant-lab-risk-permission.timer`
- `deploy/systemd/quant-lab-daily-export.timer`

Suggested production order:

- V5 telemetry sync: every 10 minutes.
- Risk permission publish: every 5 minutes, after the first telemetry sync.
- Daily expert export: after telemetry and risk permission have had time to run.

Before enabling timers, verify the commands manually with the production
`lake_root` and configured read-only API token.
