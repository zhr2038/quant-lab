from pathlib import Path

from quant_lab.data.lake import query_dataset_sql, read_parquet_dataset
from quant_lab.ingest.v5_reports import publish_v5_reports_to_lake


def test_publish_v5_reports_to_temp_lake(tmp_path):
    reports = _make_v5_reports_fixture(tmp_path)
    lake_root = tmp_path / "lake"

    result = publish_v5_reports_to_lake(reports, lake_root)

    assert result.bronze_v5_reports_rows == 1
    assert result.decision_audit_rows == 1
    assert result.strategy_run_rows == 1
    assert result.cost_bucket_daily_rows == 1
    assert result.warnings == []

    assert list((lake_root / "bronze" / "v5_reports").rglob("*.parquet"))
    assert list((lake_root / "silver" / "decision_audit").rglob("*.parquet"))
    assert list((lake_root / "silver" / "strategy_run").rglob("*.parquet"))
    assert list((lake_root / "gold" / "cost_bucket_daily").rglob("*.parquet"))

    decision_audits = read_parquet_dataset(lake_root / "silver" / "decision_audit")
    strategy_runs = read_parquet_dataset(lake_root / "silver" / "strategy_run")
    cost_buckets = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily")

    assert decision_audits.height == 1
    assert strategy_runs.height == 1
    assert cost_buckets.height == 1
    assert decision_audits["source_path"][0].endswith("decision_audit.json")
    assert strategy_runs["source_path"][0].endswith("summary.json")
    assert cost_buckets["source_path"][0].endswith("daily_cost_stats_20260216.json")
    assert cost_buckets["symbol"][0] == "BTCUSDT"
    assert cost_buckets["cost_bps"][0] == 4.2


def test_repeated_v5_publish_does_not_duplicate_logical_rows(tmp_path):
    reports = _make_v5_reports_fixture(tmp_path)
    lake_root = tmp_path / "lake"

    first = publish_v5_reports_to_lake(reports, lake_root)
    second = publish_v5_reports_to_lake(reports, lake_root)

    assert first.decision_audit_rows == 1
    assert second.decision_audit_rows == 1
    assert read_parquet_dataset(lake_root / "silver" / "decision_audit").height == 1
    assert read_parquet_dataset(lake_root / "silver" / "strategy_run").height == 1
    assert read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily").height == 1


def test_duckdb_can_query_published_dataset(tmp_path):
    reports = _make_v5_reports_fixture(tmp_path)
    lake_root = tmp_path / "lake"
    publish_v5_reports_to_lake(reports, lake_root)

    result = query_dataset_sql(
        lake_root,
        "silver/decision_audit",
        "select run_id, count(*) as row_count from dataset group by run_id",
    )

    assert result.to_dicts() == [{"run_id": "run_001", "row_count": 1}]


def _make_v5_reports_fixture(tmp_path: Path) -> Path:
    reports = tmp_path / "reports"
    (reports / "runs" / "run_001").mkdir(parents=True)
    (reports / "cost_stats").mkdir(parents=True)

    (reports / "alpha_snapshot.json").write_text('{"alpha_count": 1}', encoding="utf-8")
    (reports / "runs" / "run_001" / "decision_audit.json").write_text(
        '{"decision": "ALLOW"}', encoding="utf-8"
    )
    (reports / "runs" / "run_001" / "summary.json").write_text(
        '{"status": "complete"}', encoding="utf-8"
    )
    (reports / "cost_stats" / "daily_cost_stats_20260216.json").write_text(
        (
            '{"day": "2026-02-16", "buckets": ['
            '{"bucket_id": "btc-normal", "symbol": "BTCUSDT", '
            '"regime": "normal", "cost_bps": 4.2}'
            "]}"
        ),
        encoding="utf-8",
    )

    return reports
