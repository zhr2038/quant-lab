# Lake Optimization

quant-lab keeps high-frequency OKX WebSocket data append-only and read-only for
strategy consumers. The production lake must avoid many tiny Parquet files.

## Current Policy

High-frequency datasets use deterministic partitions:

- `bronze/okx_public_ws/day=YYYY-MM-DD/channel=<channel>/inst_id=<symbol>/`
- `silver/trade_print/day=YYYY-MM-DD/symbol=<symbol>/`
- `silver/orderbook_snapshot/day=YYYY-MM-DD/symbol=<symbol>/channel=<channel>/`

The WebSocket collector writes append-only batches. It must not rewrite the
full dataset on every message batch.

## Compaction

Run compaction periodically:

```bash
qlab compact-lake-dataset --lake-root /var/lib/quant-lab/lake --dataset okx_public_ws
qlab compact-lake-dataset --lake-root /var/lib/quant-lab/lake --dataset trade_print
qlab compact-lake-dataset --lake-root /var/lib/quant-lab/lake --dataset orderbook_snapshot
qlab lake-health --lake-root /var/lib/quant-lab/lake
```

The systemd template `quant-lab-lake-compaction.timer` runs this every 6 hours.

## Metrics

API requests are recorded in `bronze/api_request_metrics` when
`QUANT_LAB_API_METRICS_ENABLED` is true. The API records method, path, status,
duration, client host, and user agent only. It does not record authorization
headers or tokens.

Long-running CLI jobs write `gold/job_run_history` with start time, finish time,
duration, status, and redacted error text. This lets operators see which jobs
consume the most wall time.

Lake file health is written to `gold/lake_file_health_daily`.

```bash
qlab ops-summary --lake-root /var/lib/quant-lab/lake
```

## Export Rule

Daily expert export should package existing gold/report outputs. Heavy jobs
such as V5 sync, cost calibration, feature publishing, alpha evidence, and lake
compaction should run on timers before export.
