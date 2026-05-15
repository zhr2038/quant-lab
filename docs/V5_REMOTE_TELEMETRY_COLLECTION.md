# V5 Remote Telemetry Collection

V5 bundles are strategy telemetry and audit inputs. They are not the primary
market data source. `quant-lab` remains OKX-first: OKX public data and optional
OKX read-only private data provide market, fill, and fee facts. V5 bundles are
used for behavior review, audit archiving, expert review, and checking whether
V5 respected quant-lab gate and risk permission outputs.

`quant-lab` does not trade, does not mutate V5 state, and does not call trading
or withdrawal endpoints.

## Architecture

```text
qyun.hrhome.top V5 bundle
  -> quant-lab pull
  -> inbox
  -> validation
  -> secret scan/redaction
  -> restricted archive
  -> redacted archive
  -> lake bronze/silver/gold
  -> Web / expert export / risk analysis
```

Raw bundles go only to the restricted archive. The normal archive and lake must
contain redacted content only.

## Config

Example: `configs/v5_telemetry_remote.yaml`

```yaml
strategy: "v5"
remote_host: "qyun.hrhome.top"
remote_user: "v5readonly"
remote_port: 22
remote_bundle_dir: "/var/lib/v5/exports/bundles"
filename_glob: "v5_live_followup_bundle_*.tar.gz"
ssh_identity_file: "/etc/quant-lab/ssh/v5readonly_ed25519"
known_hosts_file: "/etc/quant-lab/ssh/known_hosts"
local_inbox_dir: "/var/lib/quant-lab/inbox/v5/bundles"
restricted_archive_dir: "/var/lib/quant-lab/archive_restricted/v5/bundles"
redacted_archive_dir: "/var/lib/quant-lab/archive/v5/bundles"
lake_root: "/var/lib/quant-lab/lake"
keep_remote_files: true
dry_run: false
```

Do not put passwords or key material in this file.

## Remote Permissions

On the quant-lab server:

```bash
sudo install -d -m 0700 -o quantlab -g quantlab /etc/quant-lab/ssh
sudo -u quantlab ssh-keygen -t ed25519 -f /etc/quant-lab/ssh/v5readonly_ed25519 -N ""
```

On `qyun.hrhome.top`, create a dedicated user such as `v5readonly` and install
the public key into `~v5readonly/.ssh/authorized_keys`.

Requirements:

- `v5readonly` has read-only access to the V5 bundle directory.
- Do not give `v5readonly` write permission to V5 runtime state.
- Do not use root SSH.
- Do not use password login.
- Pin `known_hosts` in `/etc/quant-lab/ssh/known_hosts`.
- Do not grant remote shell management rights unless explicitly needed.

## Manual Run

Dry-run pull:

```bash
qlab pull-v5-bundles --config /etc/quant-lab/v5_telemetry_remote.yaml --dry-run
```

End-to-end sync:

```bash
qlab sync-v5-telemetry --config /etc/quant-lab/v5_telemetry_remote.yaml
```

Validate a local bundle:

```bash
qlab validate-v5-bundle tests/fixtures/v5_live_followup_bundle_fixture.tar.gz
```

Ingest one bundle:

```bash
qlab ingest-v5-bundle \
  tests/fixtures/v5_live_followup_bundle_fixture.tar.gz \
  --lake-root /tmp/quant-lab-test-lake \
  --restricted-archive-dir /tmp/quant-lab-restricted-archive \
  --redacted-archive-dir /tmp/quant-lab-redacted-archive
```

Analyze:

```bash
qlab analyze-v5-telemetry --lake-root /tmp/quant-lab-test-lake
```

Rebuild candidate forward labels directly:

```bash
qlab build-v5-candidate-labels --lake-root /tmp/quant-lab-test-lake --date auto
qlab build-alpha-discovery-board --lake-root /tmp/quant-lab-test-lake --date auto
```

## systemd

Install timers:

```bash
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo cp deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quant-lab-v5-telemetry-sync.timer
sudo systemctl enable --now quant-lab-v5-daily-analysis.timer
```

Incremental sync runs every 3 minutes:

```text
/opt/quant-lab/.venv/bin/qlab sync-v5-telemetry --config /etc/quant-lab/v5_telemetry_remote.yaml
```

Telemetry analysis runs every 5 minutes:

```text
/opt/quant-lab/.venv/bin/qlab analyze-v5-telemetry --lake-root /var/lib/quant-lab/lake
```

## Lake Outputs

Bronze:

- `lake/bronze/strategy_telemetry/v5/bundle_manifest`
- `lake/bronze/strategy_telemetry/v5/secret_scan`
- `lake/bronze/strategy_telemetry/v5/raw_file_index`

Silver:

- `lake/silver/v5_run_summary`
- `lake/silver/v5_decision_audit`
- `lake/silver/v5_equity_point`
- `lake/silver/v5_trade_event`
- `lake/silver/v5_roundtrip`
- `lake/silver/v5_router_decision`
- `lake/silver/v5_open_position`
- `lake/silver/v5_state_snapshot`
- `lake/silver/v5_issue`
- `lake/silver/v5_config_audit`
- `lake/silver/v5_high_score_blocked_target`
- `lake/silver/v5_high_score_blocked_outcome`
- `lake/silver/v5_skipped_candidate_outcome`
- `lake/silver/v5_shadow_outcome`
- `lake/silver/v5_probe_diagnostic`
- `lake/silver/v5_quant_lab_usage`
- `lake/silver/v5_quant_lab_request`
- `lake/silver/v5_quant_lab_compliance`
- `lake/silver/v5_quant_lab_cost_usage`
- `lake/silver/v5_quant_lab_fallback`
- `lake/silver/v5_candidate_event`

Gold:

- `lake/gold/strategy_health_daily`
- `lake/gold/v5_execution_quality_daily`
- `lake/gold/v5_gate_compliance_daily`
- `lake/gold/v5_missed_opportunity_daily`
- `lake/gold/v5_config_health_daily`
- `lake/gold/v5_issue_summary_daily`
- `lake/gold/v5_quant_lab_mode_daily`
- `lake/gold/v5_quant_lab_enforcement_daily`
- `lake/gold/v5_candidate_label`
- `lake/gold/v5_candidate_quality_daily`
- `lake/gold/v5_candidate_outcome_summary`

## Candidate Snapshot Contract

V5 bundles should include `reports/candidate_snapshot.csv`. Each row is one
symbol candidate state for one V5 run. The expected columns are:

- `candidate_id`
- `run_id`
- `ts_utc`
- `symbol`
- `regime_state`
- `risk_level`
- `current_position`
- `current_weight`
- `target_weight_raw`
- `target_weight_after_risk`
- `final_score`
- `rank`
- `f1_mom_5d`
- `f2_mom_20d`
- `f3_vol_adj_ret`
- `f4_volume_expansion`
- `f5_rsi_trend_confirm`
- `alpha6_score`
- `alpha6_side`
- `ml_score`
- `mean_reversion_score`
- `expected_edge_bps`
- `required_edge_bps`
- `cost_bps`
- `cost_source`
- `eligible_before_filters`
- `final_decision`
- `block_reason`
- `strategy_candidate`

If V5 leaves `candidate_id` empty, quant-lab generates a stable id from
`run_id + symbol + strategy_candidate`.

Candidate labels are built from closed OKX `silver/market_bar` rows with a
one-bar decision delay and horizons of 4h, 8h, 12h, 24h, 48h, 72h, and 120h.
Each label row includes `gross_bps`, `net_bps_after_cost`, `mfe_bps`,
`mae_bps`, `win`, and `label_status`.

Candidate data quality is written daily:

- candidate rows by run
- feature completeness
- label completeness
- cost source coverage

Candidate outcome summaries can be grouped by `block_reason`,
`strategy_candidate`, `symbol`, and `horizon_hours`.

## quant-lab Mode Analysis

V5 telemetry may report `mode=local_only`, `shadow`, `cost_only`,
`permission_only`, or `enforce`.

- `shadow` mode records hypothetical permission violations only.
- `permission_gate_enforced=false` means a `SELL_ONLY` plus buy-side action is
  not an actual violation, but it remains visible as a hypothetical violation.
- `permission_gate_enforced=true` means a `SELL_ONLY` or `ABORT` permission plus
  risk-increasing action is an actual gate-compliance violation.

These summaries are written to:

- `lake/gold/v5_quant_lab_mode_daily`
- `lake/gold/v5_quant_lab_enforcement_daily`

## Daily Metrics

Review:

- latest bundle timestamp
- high issue count
- kill-switch state
- reconcile state
- ledger state
- auto risk level
- high-score blocked matured count
- gate compliance violations

## Security

- Raw bundles are restricted archive only.
- Redacted archive and lake must not contain secrets.
- Expert exports must not contain secrets.
- V5 cannot write quant-lab silver or gold tables.
- quant-lab only pulls bundles read-only.
- quant-lab does not trade.
