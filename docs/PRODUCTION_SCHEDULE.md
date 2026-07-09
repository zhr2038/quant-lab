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

Heavy lake maintenance runs outside the expert export path:

- Scheduled compaction must not rewrite hot OKX WebSocket partitions such as
  `bronze/okx_public_ws`, `silver/trade_print`, or `silver/orderbook_snapshot`
  while the collector is running. Those datasets are append-heavy; safe
  maintenance is limited to direct-root batch compaction that shares the
  dataset lock with the live collector.
- The default compaction service only compacts cold V5 telemetry index/usage
  datasets and records lake health. For hot market data it only compacts new
  direct batch files under the dataset root with the same dataset lock as the
  collector. Partition repair and leaf-partition compaction for hot WebSocket
  datasets are manual maintenance operations and require
  `COMPACT_HOT_WS_PARTITION_REPAIR=1`.
- `qlab lake-health` records file count, small-file ratio, and partition
  coverage in `gold/lake_file_health_daily`.
- `qlab ops-summary` reads API request metrics, job durations, and lake file
  health without triggering research recomputation.

The daily export service should package the latest already-computed outputs. It
must not be used as the normal way to run V5 sync, cost calibration, feature
publication, or alpha evidence generation.

Additional V5 telemetry analysis and candidate research jobs are intentionally
split:

- High-frequency `qlab sync-v5-telemetry` ingests at most one new bundle only.
  It skips historical outcome files and does not run V5 health analysis or
  rebuild candidate research outputs.
- Frequent `qlab analyze-v5-telemetry --skip-candidate-gold` refreshes V5 health
  gold tables without touching candidate labels, strategy evidence, or the
  alpha discovery board. It should skip rather than overlap when a heavy
  research refresh is already running.
- Hourly `quant-lab-v5-research-refresh.service` runs incremental
  candidate research only:
  `qlab build-v5-candidate-labels --mode incremental --lookback-days 8`,
  `qlab build-strategy-evidence --mode incremental --lookback-days 8 --skip-historical-outcomes`,
  and
  `qlab build-alpha-discovery-board --skip-legacy-outcome-counts`.
  It consumes recently ingested raw inputs and previously computed
  `gold/strategy_evidence_sample` state. It must not rescan all historical
  shadow/blocked outcome files on every run.
- `quant-lab-v5-regime-router.service` runs `qlab build-regime-router`
  separately from the heavier research refresh. Keep it independent so
  `gold/market_regime_daily`, `gold/strategy_regime_matrix`, and
  `gold/regime_strategy_advisory` do not go stale when alpha-factory or entry
  quality refreshes consume the research-refresh time budget.
- `qlab build-alpha-evidence` remains separate from candidate board refresh.

Historical full rebuilds are manual maintenance actions only. Use `--mode full`
or `--include-legacy-outcome-counts` after a schema/rule migration or a known
backfill correction, then monitor memory and runtime explicitly.

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

Cost calibration has the same dependency caveat: `After=quant-lab-okx-readonly-backfill.service`
does not start the read-only private backfill when independent timers drift.
The `quant-lab-cost-calibration.service` template therefore starts
`quant-lab-okx-readonly-backfill.service` in `ExecStartPre` when
`/etc/quant-lab/okx_readonly.env` exists. When that optional environment file is
absent, the service prints `SKIP_OKX_READONLY_BACKFILL_ENV_MISSING` and continues
with public proxy calibration instead of silently pretending actual/mixed cost
coverage is available.

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
- `deploy/systemd/quant-lab-v5-research-refresh.timer`
- `deploy/systemd/quant-lab-v5-regime-router.timer`
- `deploy/systemd/quant-lab-alpha-evidence.timer`
- `deploy/systemd/quant-lab-risk-permission.timer`
- `deploy/systemd/quant-lab-web-v2-smoke.timer`
- `deploy/systemd/quant-lab-daily-export.timer`

Suggested production order:

- V5 telemetry sync: every 10 minutes, one newest bundle per run.
- V5 telemetry analysis: every 30 minutes, using `--skip-candidate-gold`.
- Candidate labels, strategy evidence, alpha factory, entry-quality summaries,
  paper tracking, and alpha discovery board: every 1 hour. This cadence keeps
  `strategy_opportunity_advisory` below V5's 90 minute freshness window without
  making daily expert export rescan historical research state.
- Regime router: every 30 minutes, as an independent short job.
- Alpha evidence and gate publishing: every 15 minutes.
- Risk permission publish: every 10 minutes, after telemetry and gate refresh
  and no less frequently than every 30 minutes.
- Web V2/API smoke: every 10 minutes, using the production API token from
  `/etc/quant-lab/quant_lab_api.env`. It checks `/web-v2`, no-store snapshot
  headers, protected API contracts, cost-estimate trust flags, and risk
  permission remains non-live.
- Daily expert export: after telemetry and risk permission have had time to run;
  the export should still run `--pre-export-v5-refresh --allow-stale-v5` so the
  latest pack records current V5 snapshot provenance instead of a non-authoritative
  packaging-only snapshot.

`qlab build-strategy-evidence` remains available for offline legacy telemetry
research, but production readiness and daily exports should use
`gold/alpha_discovery_board` as the primary candidate decision board.

Before enabling timers, verify the commands manually with the production
`lake_root` and configured read-only API token.
