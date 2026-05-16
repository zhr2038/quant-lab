# Production Schedule

`quant-lab` production jobs are read-only research jobs. They must not place,
cancel, amend, transfer, withdraw, or mutate exchange state.

## Required Ordering

Daily and recurring jobs must keep this order:

1. `qlab sync-v5-telemetry`
2. `qlab calibrate-costs`
3. `qlab build-alpha-evidence`
4. `qlab publish-gate-decisions`
5. `qlab publish-risk-permission`
6. `qlab export-daily`

Additional V5 telemetry analysis and candidate research jobs should run before
`publish-risk-permission` when they are enabled:

- `qlab analyze-v5-telemetry`
- `qlab build-v5-candidate-labels`
- `qlab build-alpha-discovery-board`
- `qlab build-strategy-evidence`

The ordering matters because `risk_permission` must be evaluated against the
latest V5 telemetry, gate, cost, and data-health state. The daily export service
must force a final `qlab publish-risk-permission` immediately before
`qlab export-daily`; `systemd After=` ordering alone is not enough because it
does not start the dependency on demand. The repository service template uses:

```text
ExecStartPre=/opt/quant-lab/.venv/bin/qlab publish-risk-permission --lake-root /var/lib/quant-lab/lake --strategy v5 --version 5.0.0
```

If `risk_permission.as_of_ts` is older than the latest V5 bundle timestamp, the
expert export reports:

```text
risk_permission_stale_vs_v5_telemetry
```

If `risk_permission.expires_at` is earlier than the export timestamp, the
expert export reports a critical check:

```text
risk_permission_not_expired_at_export
```

In that case rerun:

```bash
qlab publish-risk-permission --lake-root /var/lib/quant-lab/lake --strategy v5 --version 5.0.0
qlab export-daily --date "$(date -u +%F)" --lake-root /var/lib/quant-lab/lake --out-dir /var/lib/quant-lab/exports
```

Production `risk_permission` rows use a 90 minute validity window. The
`quant-lab-risk-permission.timer` must therefore run at least every 30 minutes;
the repository template runs it every 10 minutes so a missed single run does not
leave V5 reading expired permission for hours. If the API sees only expired
published rows, `/v1/risk/live-permission` returns `NO_FRESH_PERMISSION` instead
of repeatedly returning `EXPIRED_*`.

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
- `deploy/systemd/quant-lab-v5-daily-analysis.timer`
- `deploy/systemd/quant-lab-alpha-evidence.timer`
- `deploy/systemd/quant-lab-risk-permission.timer`
- `deploy/systemd/quant-lab-daily-export.timer`

Suggested production order:

- V5 telemetry sync: every 3 minutes.
- V5 telemetry analysis: every 5 minutes.
- Candidate labels, alpha discovery board, alpha evidence, and gate publishing:
  every 15 minutes.
- Risk permission publish: every 10 minutes, after telemetry and gate refresh
  and no less frequently than every 30 minutes.
- Daily expert export: after telemetry and risk permission have had time to run.

`qlab build-strategy-evidence` remains available for offline legacy telemetry
research, but production readiness and daily exports should use
`gold/alpha_discovery_board` as the primary candidate decision board.

Before enabling timers, verify the commands manually with the production
`lake_root` and configured read-only API token.
