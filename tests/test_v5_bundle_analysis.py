from datetime import UTC, datetime

import polars as pl

import quant_lab.strategy_telemetry.analyze as analyze_module
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.strategy_telemetry.analyze import analyze_v5_telemetry
from quant_lab.strategy_telemetry.ingest import ingest_v5_bundle
from tests.v5_bundle_fixture import make_v5_bundle_fixture


def test_analyze_flags_kill_switch_critical(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_state(lake, "kill_switch", {"enabled": True})

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.status == "CRITICAL"
    assert "kill_switch_enabled" in result.critical_reasons


def test_analyze_flags_reconcile_failure(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_state(lake, "reconcile_status", {"ok": False})

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.status == "CRITICAL"
    assert "reconcile_not_ok" in result.critical_reasons


def test_analyze_flags_missing_recent_bundle(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake, bundle_ts=datetime(2026, 5, 8, tzinfo=UTC))

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.status == "WARNING"
    assert "latest bundle is older than 24 hours" in result.warnings


def test_analyze_can_skip_candidate_gold_refresh(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    called: list[str] = []

    monkeypatch.setattr(
        analyze_module,
        "_build_candidate_labels_safely",
        lambda *_args, **_kwargs: called.append("labels"),
    )
    monkeypatch.setattr(
        analyze_module,
        "_build_alpha_discovery_board_safely",
        lambda *_args, **_kwargs: called.append("board"),
    )
    monkeypatch.setattr(
        analyze_module,
        "_build_strategy_evidence_safely",
        lambda *_args, **_kwargs: called.append("evidence"),
    )

    result = analyze_v5_telemetry(
        lake,
        date="2026-05-10",
        refresh_candidate_gold=False,
    )

    assert result.strategy == "v5"
    assert called == []


def test_analyze_detects_high_score_blocked_issue(tmp_path):
    bundle = make_v5_bundle_fixture(tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz")
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    assert result.validation.valid is True
    analysis = analyze_v5_telemetry(lake, date="2026-05-10")
    assert "review high_score_blocked_matured_without_label" in analysis.next_actions


def test_analyze_uses_window_summary_metrics_before_row_counts(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                _run_summary_row(
                    "summaries/window_summary.json",
                    {
                        "run_count": 4,
                        "recent_24h_decision_audit_count": 5,
                        "latest_24h_trade_count": 6,
                        "last_72h_trade_count": 7,
                        "last_72h_roundtrip_count": 8,
                        "open_position_count": 9,
                        "dust_residual_position_count": 2,
                        "high_issue_count": 3,
                        "medium_issue_count": 1,
                    },
                ),
                _run_summary_row("raw/recent_runs/run_001/summary.json", {"run_id": "run_001"}),
            ]
        ),
        lake / "silver/v5_run_summary",
    )
    write_parquet_dataset(
        pl.DataFrame([_event_row("old") for _ in range(20)]),
        lake / "silver/v5_decision_audit",
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.run_count_72h == 4
    assert result.decision_audit_count_24h == 5
    assert result.trade_count_24h == 6
    assert result.trade_count_72h == 7
    assert result.roundtrip_count_72h == 8
    assert result.open_position_count == 9
    assert result.dust_residual_position_count == 2
    assert result.high_issue_count == 3
    assert result.medium_issue_count == 1


def test_analyze_counts_recent_rows_by_bundle_ts_when_window_summary_missing(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                _event_row("recent", bundle_ts=datetime(2026, 5, 10, 20, tzinfo=UTC)),
                _event_row("old", bundle_ts=datetime(2026, 5, 8, 20, tzinfo=UTC)),
            ]
        ),
        lake / "silver/v5_trade_event",
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.trade_count_24h == 1
    assert result.trade_count_72h == 2


def test_analyze_reads_auto_risk_current_level(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_state(lake, "auto_risk_eval", {"current_level": "ELEVATED"})

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.auto_risk_level == "ELEVATED"


def test_config_not_consumed_count_uses_unique_keys(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                _config_row("alpha_threshold", status="not_consumed"),
                _config_row("alpha_threshold", status="not_consumed"),
                _config_row("risk.max_weight", consumed=False),
                _config_row("risk.min_score", status="consumed", consumed=True),
            ]
        ),
        lake / "silver/v5_config_audit",
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.config_not_consumed_count == 2
    assert result.config_not_consumed_count_unknown is False
    assert result.config_not_consumed_top_keys == ["alpha_threshold", "risk.max_weight"]


def test_config_not_consumed_count_unknown_when_unparseable(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    write_parquet_dataset(
        pl.DataFrame([{"blob": "not_consumed not_consumed not_consumed"}]),
        lake / "silver/v5_config_audit",
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.config_not_consumed_count == 0
    assert result.config_not_consumed_count_unknown is True
    assert "config_not_consumed_count_unknown" in result.warnings


def test_config_not_consumed_count_uses_vectorized_raw_markers(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {"key": "alpha_threshold", "raw_payload_json": '{"status":"not_consumed"}'},
                {"key": "alpha_threshold", "raw_payload_json": '{"status":"not_consumed"}'},
                {"config_key": "risk.max_weight", "consumed": "false"},
                {"path": "risk.min_score", "status": "consumed", "consumed": "true"},
            ]
        ),
        lake / "silver/v5_config_audit",
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.config_not_consumed_count == 2
    assert result.config_not_consumed_count_unknown is False
    assert result.config_not_consumed_top_keys == ["alpha_threshold", "risk.max_weight"]


def test_high_score_blocked_next_action_matches_matured_count(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    **_event_row("issue"),
                    "severity": "high",
                    "issue_type": "high_score_blocked_matured_without_label",
                    "message": "needs label",
                }
            ]
        ),
        lake / "silver/v5_issue",
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert "review high_score_blocked_matured_without_label" in result.next_actions
    assert result.high_score_blocked_matured_count == 1


def test_matured_and_profitable_counts_handle_string_flags(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {"matured": "true", "profitable": "true", **_event_row("outcome_1")},
                {"matured": "1", "profitable": "false", **_event_row("outcome_2")},
                {"matured": "false", "profitable": "yes", **_event_row("outcome_3")},
            ]
        ),
        lake / "silver/v5_high_score_blocked_outcome",
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.high_score_blocked_matured_count == 2
    assert result.high_score_blocked_profitable_count == 2


def test_matured_and_profitable_counts_ignore_time_only_rows(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {"bundle_ts": datetime(2026, 5, 10, tzinfo=UTC)},
                {"bundle_ts": datetime(2026, 5, 10, tzinfo=UTC)},
            ]
        ),
        lake / "silver/v5_high_score_blocked_outcome",
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.high_score_blocked_matured_count == 0
    assert result.high_score_blocked_profitable_count == 0


def test_quant_lab_shadow_mode_marks_hypothetical_not_actual_violation(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_quant_lab_usage(lake, mode="shadow", enforced=False)
    _write_quant_lab_compliance(lake, permission="SELL_ONLY", side="buy")

    result = analyze_v5_telemetry(lake, date="2026-05-10")
    enforcement = read_parquet_dataset(lake / "gold/v5_quant_lab_enforcement_daily")

    assert result.quant_lab_mode == "shadow"
    assert result.quant_lab_actual_violation_count == 0
    assert result.quant_lab_hypothetical_violation_count == 1
    assert "gate_compliance_violation" not in result.critical_reasons
    assert enforcement["actual_violation_count"][0] == 0
    assert enforcement["hypothetical_violation_count"][0] == 1


def test_quant_lab_enforce_mode_marks_sell_only_buy_actual_violation(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_quant_lab_usage(lake, mode="enforce", enforced=True)
    _write_quant_lab_compliance(lake, permission="SELL_ONLY", side="buy")

    result = analyze_v5_telemetry(lake, date="2026-05-10")
    mode = read_parquet_dataset(lake / "gold/v5_quant_lab_mode_daily")

    assert result.status == "CRITICAL"
    assert result.quant_lab_actual_violation_count == 1
    assert "gate_compliance_violation" in result.critical_reasons
    assert mode["mode"][0] == "enforce"
    assert mode["permission_gate_enforced"][0] == True  # noqa: E712


def test_quant_lab_compliance_prefers_explicit_violation_fields(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_quant_lab_usage(lake, mode="enforce", enforced=True)
    _write_quant_lab_compliance(
        lake,
        permission="ALLOW",
        side="sell",
        actual_violation=True,
        hypothetical_violation=False,
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.quant_lab_actual_violation_count == 1
    assert result.quant_lab_hypothetical_violation_count == 0


def test_quant_lab_cost_only_does_not_create_permission_actual_violation(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_quant_lab_usage(lake, mode="cost_only", enforced=True)
    _write_quant_lab_compliance(lake, permission="SELL_ONLY", side="buy", new_risk=True)

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.quant_lab_actual_violation_count == 0
    assert result.quant_lab_hypothetical_violation_count == 1


def test_quant_lab_reduce_only_buy_is_not_risk_increasing_violation(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_quant_lab_usage(lake, mode="permission_only", enforced=True)
    _write_quant_lab_compliance(
        lake,
        permission="SELL_ONLY",
        side="buy",
        reduce_only=True,
        new_risk=True,
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.quant_lab_actual_violation_count == 0
    assert result.quant_lab_hypothetical_violation_count == 0


def test_quant_lab_compliance_uses_production_permission_fields(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_quant_lab_usage(lake, mode="permission_only", enforced=True)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    **_event_row("production-compliance"),
                    "source_path_inside_bundle": "summaries/quant_lab_compliance.csv",
                    "mode": "permission_only",
                    "permission_gate_enforced": "true",
                    "effective_permission_decision": "SELL_ONLY",
                    "intent": "OPEN_LONG",
                    "side": "buy",
                }
            ]
        ),
        lake / "silver/v5_quant_lab_compliance",
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")

    assert result.quant_lab_actual_violation_count == 1
    assert result.quant_lab_hypothetical_violation_count == 0
    assert "gate_compliance_violation" in result.critical_reasons


def test_quant_lab_mode_can_parse_decision_audit_nested_quant_lab(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    **_event_row("decision_audit"),
                    "source_path_inside_bundle": "raw/recent_runs/run_001/decision_audit.json",
                    "raw_payload_json": __import__("json").dumps(
                        {
                            "quant_lab": {
                                "mode": "permission_only",
                                "apply_permission_gate": True,
                                "apply_cost_gate": False,
                                "strategy_version": "5.0.0",
                            }
                        }
                    ),
                }
            ]
        ),
        lake / "silver/v5_decision_audit",
    )

    result = analyze_v5_telemetry(lake, date="2026-05-10")
    mode = read_parquet_dataset(lake / "gold/v5_quant_lab_mode_daily")

    assert result.quant_lab_mode == "permission_only"
    assert result.permission_gate_enforced is True
    assert result.cost_gate_enforced is False
    assert mode["mode"][0] == "permission_only"


def _write_manifest(lake, bundle_ts=datetime(2026, 5, 10, 14, 2, 49, tzinfo=UTC)):
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": "abc",
                    "bundle_name": "v5_live_followup_bundle_20260510T140249Z.tar.gz",
                    "bundle_ts": bundle_ts,
                    "ingest_ts": datetime(2026, 5, 10, 14, 3, tzinfo=UTC),
                    "schema_version": "test",
                }
            ]
        ),
        lake / "bronze/strategy_telemetry/v5/bundle_manifest",
    )


def _write_state(lake, state_type, payload):
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": "abc",
                    "bundle_name": "bundle.tar.gz",
                    "bundle_ts": datetime(2026, 5, 10, tzinfo=UTC),
                    "ingest_ts": datetime(2026, 5, 10, tzinfo=UTC),
                    "schema_version": "test",
                    "source_path_inside_bundle": f"raw/state/{state_type}.json",
                    "run_id": None,
                    "row_index": 0,
                    "state_type": state_type,
                    "ok": payload.get("ok"),
                    "enabled": payload.get("enabled"),
                    "level": payload.get("current_level") or payload.get("level", ""),
                    "raw_payload_json": __import__("json").dumps(payload),
                }
            ]
        ),
        lake / "silver/v5_state_snapshot",
    )


def test_analysis_writes_gold(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    result = analyze_v5_telemetry(lake, date="2026-05-10")
    health = read_parquet_dataset(lake / "gold/strategy_health_daily")

    assert result.date == "2026-05-10"
    assert health.height == 1


def test_analysis_publishes_strategy_evidence_summary(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    _write_candidate_label(lake)

    analyze_v5_telemetry(lake, date="2026-05-10")

    evidence = read_parquet_dataset(lake / "gold/strategy_evidence")
    assert evidence.height > 0
    assert {"strategy_candidate", "symbol", "regime_state", "horizon_hours"}.issubset(
        evidence.columns
    )


def test_analyze_warns_on_consecutive_expired_remote_permissions(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    rows = []
    for index, status in enumerate(["EXPIRED_ABORT", "EXPIRED_ABORT"]):
        rows.append(
            {
                **_event_row(f"permission-{index}"),
                "source_path_inside_bundle": "raw/reports/quant_lab_requests.jsonl",
                "event_id": f"permission-{index}",
                "ts_utc": datetime(2026, 5, 10, 12, index, tzinfo=UTC),
                "path": "/v1/risk/live-permission",
                "status_code": 200,
                "success": True,
                "fallback_used": False,
                "permission_status": status,
                "raw_payload_json": __import__("json").dumps(
                    {
                        "event_id": f"permission-{index}",
                        "ts_utc": f"2026-05-10T12:0{index}:00Z",
                        "path": "/v1/risk/live-permission",
                        "status_code": 200,
                        "success": True,
                        "fallback_used": False,
                        "permission_status": status,
                    }
                ),
            }
        )
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver/v5_quant_lab_request")

    result = analyze_v5_telemetry(lake, date="2026-05-10")
    health = read_parquet_dataset(lake / "gold/strategy_health_daily")

    assert result.latest_permission_status == "EXPIRED_ABORT"
    assert result.stale_permission_consecutive_count == 2
    assert any(
        "stale or expired remote permission repeated 2 times" in item
        for item in result.warnings
    )
    assert "run quant-lab publish-risk-permission and verify risk timer" in result.next_actions
    assert health["stale_permission_consecutive_count"][0] == 2


def test_analyze_deduplicates_request_and_summary_fallback_events(tmp_path):
    lake = tmp_path / "lake"
    _write_manifest(lake)
    request_row = {
        **_event_row("request-timeout"),
        "source_path_inside_bundle": "raw/reports/quant_lab_requests.jsonl",
        "run_id": "20260512_23",
        "ts_utc": "2026-05-12T15:00:26.841187Z",
        "endpoint_path": "/v1/risk/live-permission",
        "event_type": "request",
        "status_code": "not_observable",
        "success": "false",
        "fallback_used": True,
        "error_type": "QuantLabTimeout",
        "error": "QuantLabTimeout",
        "raw_payload_json": (
            '{"event_type":"request","run_id":"20260512_23",'
            '"ts":"2026-05-12T15:00:26.841187Z",'
            '"endpoint_path":"/v1/risk/live-permission",'
            '"status_code":null,"success":false,"fallback_used":true,'
            '"error_type":"QuantLabTimeout"}'
        ),
    }
    fallback_row = {
        **_event_row("summary-timeout"),
        "source_path_inside_bundle": "summaries/quant_lab_fallbacks.csv",
        "run_id": "20260512_23",
        "ts_utc": "2026-05-12T15:00:26.841187Z",
        "endpoint_path": "/v1/risk/live-permission",
        "event_type": "fallback",
        "status_code": "not_observable",
        "success": "false",
        "fallback_used": True,
        "error_type": "QuantLabTimeout",
        "error": "QuantLabTimeout",
        "diagnosis": "fallback_request",
        "raw_payload_json": (
            '{"event_type":"fallback","run_id":"20260512_23",'
            '"ts":"2026-05-12T15:00:26.841187Z",'
            '"endpoint_path":"/v1/risk/live-permission",'
            '"status_code":null,"success":false,"fallback_used":true,'
            '"error_type":"QuantLabTimeout"}'
        ),
    }
    write_parquet_dataset(pl.DataFrame([request_row]), lake / "silver/v5_quant_lab_request")
    write_parquet_dataset(pl.DataFrame([fallback_row]), lake / "silver/v5_quant_lab_fallback")

    result = analyze_v5_telemetry(lake, date="2026-05-12")
    health = read_parquet_dataset(lake / "gold/strategy_health_daily")

    assert result.unique_request_count == 1
    assert result.unique_actual_fallback_count == 1
    assert result.actual_fallback_count == 1
    assert result.unique_event_rows == 1
    assert result.duplicate_event_rows == 1
    assert health["actual_fallback_count"][0] == 1


def _run_summary_row(source_path, payload):
    return {
        "strategy": "v5",
        "bundle_sha256": "abc",
        "bundle_name": "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        "bundle_ts": datetime(2026, 5, 10, 14, 2, 49, tzinfo=UTC),
        "ingest_ts": datetime(2026, 5, 10, 14, 3, tzinfo=UTC),
        "schema_version": "test",
        "source_path_inside_bundle": source_path,
        "run_id": "run_001",
        "row_index": 0,
        "raw_payload_json": __import__("json").dumps(payload),
    }


def _event_row(label, bundle_ts=datetime(2026, 5, 10, 14, 2, 49, tzinfo=UTC)):
    return {
        "strategy": "v5",
        "bundle_sha256": f"abc-{label}",
        "bundle_name": f"{label}.tar.gz",
        "bundle_ts": bundle_ts,
        "ingest_ts": bundle_ts,
        "schema_version": "test",
        "source_path_inside_bundle": f"raw/recent_runs/{label}/decision_audit.json",
        "run_id": label,
        "row_index": 0,
        "raw_payload_json": "{}",
    }


def _write_candidate_label(lake):
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "candidate_label_schema_version": "test",
                    "candidate_id": "candidate-1",
                    "run_id": "run-1",
                    "ts_utc": datetime(2026, 5, 10, 12, tzinfo=UTC),
                    "symbol": "BNB-USDT",
                    "strategy_candidate": "v5.alt_impulse_shadow",
                    "block_reason": "shadow",
                    "final_decision": "SHADOW",
                    "horizon_hours": 24,
                    "decision_ts": datetime(2026, 5, 10, 13, tzinfo=UTC),
                    "label_ts": datetime(2026, 5, 11, 13, tzinfo=UTC),
                    "entry_close": 100.0,
                    "label_close": 101.0,
                    "gross_bps": 100.0,
                    "net_bps_after_cost": 90.0,
                    "mfe_bps": 120.0,
                    "mae_bps": -20.0,
                    "win": True,
                    "label_status": "complete",
                    "label_reason": "",
                    "cost_bps": 10.0,
                    "cost_source": "mixed_actual_proxy",
                    "alpha6_side": "long",
                    "regime_state": "trend",
                    "risk_level": "normal",
                    "protect_level": "",
                    "final_score": 0.7,
                    "expected_edge_bps": 120.0,
                    "required_edge_bps": 20.0,
                    "source_event_bundle_sha256": "abc",
                    "source_path_inside_bundle": "raw/candidate_events.jsonl",
                    "created_at": datetime(2026, 5, 10, 12, 1, tzinfo=UTC),
                    "source": "test",
                }
            ]
        ),
        lake / "gold/v5_candidate_label",
    )


def _config_row(key, *, status="", consumed=None):
    return {
        "strategy": "v5",
        "bundle_sha256": "abc",
        "bundle_name": "bundle.tar.gz",
        "bundle_ts": datetime(2026, 5, 10, 14, 2, 49, tzinfo=UTC),
        "ingest_ts": datetime(2026, 5, 10, 14, 3, tzinfo=UTC),
        "schema_version": "test",
        "source_path_inside_bundle": "summaries/config_runtime_consumption_audit.csv",
        "run_id": None,
        "row_index": 0,
        "key": key,
        "status": status,
        "consumed": consumed,
        "raw_payload_json": "{}",
    }


def _write_quant_lab_usage(lake, *, mode: str, enforced: bool) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    **_event_row("quant_lab_usage"),
                    "source_path_inside_bundle": "raw/quant_lab/quant_lab_usage.jsonl",
                    "mode": mode,
                    "permission_gate_enforced": enforced,
                    "raw_payload_json": __import__("json").dumps(
                        {"mode": mode, "permission_gate_enforced": enforced}
                    ),
                }
            ]
        ),
        lake / "silver/v5_quant_lab_usage",
    )


def _write_quant_lab_compliance(
    lake,
    *,
    permission: str,
    side: str,
    actual_violation: bool | None = None,
    hypothetical_violation: bool | None = None,
    reduce_only: bool | None = None,
    new_risk: bool | None = None,
) -> None:
    payload = {"permission": permission, "side": side}
    row = {
        **_event_row("quant_lab_compliance"),
        "source_path_inside_bundle": "summaries/quant_lab_compliance.csv",
        "permission": permission,
        "side": side,
    }
    for key, value in {
        "actual_violation": actual_violation,
        "hypothetical_violation": hypothetical_violation,
        "reduce_only": reduce_only,
        "new_risk": new_risk,
    }.items():
        if value is not None:
            row[key] = value
            payload[key] = value
    row["raw_payload_json"] = __import__("json").dumps(payload)
    write_parquet_dataset(
        pl.DataFrame([row]),
        lake / "silver/v5_quant_lab_compliance",
    )
