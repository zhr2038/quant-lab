from pathlib import Path

from quant_lab.ops.dataset_registry import (
    dataset_names,
    dataset_path_map,
    dataset_registry_rows,
    get_dataset_spec,
)
from quant_lab.research.publish import ALPHA_EVIDENCE_SCHEMA, GATE_DECISION_SCHEMA

KNOWN_DATASET_SCHEMAS = {
    "alpha_evidence": set(ALPHA_EVIDENCE_SCHEMA),
    "gate_decision": set(GATE_DECISION_SCHEMA),
}


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
    assert paths["cost_bootstrap_readiness"] == Path("gold") / "cost_bootstrap_readiness"
    assert paths["v5_candidate_event"] == Path("silver") / "v5_candidate_event"


def test_dataset_registry_covers_v5_and_research_governance_datasets():
    names = set(dataset_names())
    expected = {
        "fill_event",
        "account_bill",
        "order_event",
        "v5_decision_audit",
        "v5_trade_event",
        "v5_btc_probe_entry_quality_audit",
        "v5_quant_lab_request",
        "v5_quant_lab_cost_usage",
        "v5_quant_lab_fallback",
        "v5_quant_lab_compliance",
        "v5_cost_probe_p3_preflight",
        "v5_cost_probe_live_execution_status",
        "v5_cost_probe_order_event",
        "v5_cost_probe_roundtrip_event",
        "v5_candidate_label",
        "v5_shadow_outcome",
        "v5_paper_strategy_run",
        "v5_paper_strategy_proposal_ack",
        "v5_paper_strategy_daily",
        "v5_paper_slippage_coverage",
        "paper_strategy_registry",
        "paper_strategy_promotion_gate",
        "strategy_evidence",
        "strategy_evidence_sample",
        "strategy_evidence_quality",
        "alpha_discovery_board",
        "v5_execution_quality_daily",
        "v5_gate_compliance_daily",
        "v5_missed_opportunity_daily",
        "v5_config_health_daily",
        "v5_issue_summary_daily",
    }

    assert expected.issubset(names)


def test_dataset_registry_rows_are_serializable():
    rows = dataset_registry_rows()
    market_bar = next(row for row in rows if row["dataset_id"] == "market_bar")

    assert market_bar["owner"] == "market-data"
    assert "primary_key_json" in market_bar
    assert "quality_rules_json" in market_bar


def test_research_dataset_required_columns_match_published_schemas():
    alpha_evidence = get_dataset_spec("alpha_evidence")
    gate_decision = get_dataset_spec("gate_decision")

    assert alpha_evidence is not None
    assert set(alpha_evidence.required_columns).issubset(ALPHA_EVIDENCE_SCHEMA)
    assert "status" not in alpha_evidence.required_columns
    assert "sample_count" not in alpha_evidence.required_columns

    assert gate_decision is not None
    assert set(gate_decision.required_columns).issubset(GATE_DECISION_SCHEMA)
    assert "decision" not in gate_decision.required_columns


def test_dataset_registry_primary_keys_are_declared_schema_subsets():
    bad_specs: dict[str, tuple[str, ...]] = {}
    for name in dataset_names():
        spec = get_dataset_spec(name)
        assert spec is not None
        if not spec.primary_key:
            continue
        allowed_columns = set(spec.required_columns) | KNOWN_DATASET_SCHEMAS.get(name, set())
        missing = tuple(column for column in spec.primary_key if column not in allowed_columns)
        if missing:
            bad_specs[name] = missing

    assert bad_specs == {}


def test_alpha_evidence_primary_key_matches_real_schema():
    alpha_evidence = get_dataset_spec("alpha_evidence")

    assert alpha_evidence is not None
    assert set(alpha_evidence.primary_key).issubset(ALPHA_EVIDENCE_SCHEMA)
    assert "symbol" not in alpha_evidence.primary_key
    assert "horizon_hours" not in alpha_evidence.primary_key


def test_job_run_history_uses_duration_seconds_schema():
    job_run_history = get_dataset_spec("job_run_history")

    assert job_run_history is not None
    assert "duration_seconds" in job_run_history.required_columns
    assert "duration_s" not in job_run_history.required_columns


def test_v5_registry_matches_current_envelope_and_daily_schemas():
    decision_audit = get_dataset_spec("v5_decision_audit")
    usage = get_dataset_spec("v5_quant_lab_usage")
    cost_usage = get_dataset_spec("v5_quant_lab_cost_usage")
    trade_event = get_dataset_spec("v5_trade_event")
    strategy_health = get_dataset_spec("strategy_health_daily")
    strategy_evidence_quality = get_dataset_spec("strategy_evidence_quality")

    assert decision_audit is not None
    assert decision_audit.required_columns == (
        "bundle_ts",
        "ingest_ts",
        "run_id",
        "raw_payload_json",
    )
    assert decision_audit.timestamp_column == "ingest_ts"

    assert usage is not None
    assert usage.required_columns == ("bundle_ts", "ingest_ts", "raw_payload_json")
    assert usage.timestamp_column == "ingest_ts"

    assert cost_usage is not None
    assert "freshness" not in cost_usage.quality_rules
    assert trade_event is not None
    assert "freshness" not in trade_event.quality_rules
    btc_probe = get_dataset_spec("v5_btc_probe_entry_quality_audit")
    assert btc_probe is not None
    assert btc_probe.required_columns == (
        "bundle_ts",
        "ingest_ts",
        "source_path_inside_bundle",
        "same_symbol_reentry_bypass",
        "anti_chase_flag",
        "raw_payload_json",
    )
    assert btc_probe.timestamp_column == "ingest_ts"

    p3_preflight = get_dataset_spec("v5_cost_probe_p3_preflight")
    assert p3_preflight is not None
    assert "approved_live_order_execution" in p3_preflight.required_columns
    assert "live_order_effect" in p3_preflight.required_columns
    assert p3_preflight.timestamp_column == "ingest_ts"

    live_status = get_dataset_spec("v5_cost_probe_live_execution_status")
    assert live_status is not None
    assert "authorization_fresh" in live_status.required_columns
    assert "recovery_required" in live_status.required_columns
    assert live_status.timestamp_column == "generated_at_utc"

    order_events = get_dataset_spec("v5_cost_probe_order_event")
    assert order_events is not None
    assert order_events.primary_key == ("event_key",)
    assert "order_key" in order_events.required_columns
    assert order_events.timestamp_column == "event_ts"

    roundtrip_events = get_dataset_spec("v5_cost_probe_roundtrip_event")
    assert roundtrip_events is not None
    assert roundtrip_events.primary_key == ("event_key",)
    assert "roundtrip_key" in roundtrip_events.required_columns
    assert roundtrip_events.timestamp_column == "event_ts"

    assert strategy_health is not None
    assert strategy_health.required_columns == ("date", "status", "latest_bundle_ts")
    assert strategy_health.timestamp_column == "latest_bundle_ts"

    assert strategy_evidence_quality is not None
    assert strategy_evidence_quality.required_columns == (
        "severity",
        "warning_type",
        "warning_count",
        "created_at",
    )
