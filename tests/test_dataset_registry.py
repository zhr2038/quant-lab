from pathlib import Path

from quant_lab.ops.dataset_registry import (
    dataset_names,
    dataset_path_map,
    dataset_registry_rows,
    get_dataset_spec,
)


def test_dataset_registry_declares_core_ownership_and_sla():
    market_bar = get_dataset_spec("market_bar")
    risk_permission = get_dataset_spec("risk_permission")

    assert market_bar is not None
    assert market_bar.layer == "silver"
    assert market_bar.owner == "market-data"
    assert market_bar.primary_key == ("venue", "symbol", "timeframe", "ts")
    assert market_bar.closed_bar_column == "is_closed"
    assert market_bar.freshness_seconds is not None

    assert risk_permission is not None
    assert risk_permission.owner == "risk"
    assert risk_permission.freshness_seconds == 90 * 60


def test_dataset_registry_exposes_paths_for_api_and_lake_health():
    paths = dataset_path_map()

    assert "market_bar" in dataset_names()
    assert paths["market_bar"] == Path("silver") / "market_bar"
    assert paths["cost_bucket_daily"] == Path("gold") / "cost_bucket_daily"
    assert paths["v5_candidate_event"] == Path("silver") / "v5_candidate_event"


def test_dataset_registry_rows_are_serializable():
    rows = dataset_registry_rows()
    market_bar = next(row for row in rows if row["dataset_id"] == "market_bar")

    assert market_bar["owner"] == "market-data"
    assert "primary_key_json" in market_bar
    assert "quality_rules_json" in market_bar
