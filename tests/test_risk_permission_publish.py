import json
from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.contracts.models import (
    GateDecision,
    GateStatus,
    RiskAction,
    RiskPermission,
    RiskPermissionStatus,
)
from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.risk.publish import _dependency_source_sha, lake_data_health, publish_risk_permission


def test_publish_risk_permission_sell_only_when_cost_missing(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_fresh_market_bar(lake)

    result = publish_risk_permission(lake, strategy="v5", version="5.0.0")

    assert result.permission == "SELL_ONLY"
    assert result.reasons == ["cost_health_missing"]


def test_publish_risk_permission_allows_paper_for_paper_ready_gate(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.PAPER_READY)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)

    publish_risk_permission(lake, strategy="v5", version="5.0.0")
    permission = read_parquet_dataset(lake / "gold" / "risk_permission").to_dicts()[0]

    assert permission["permission"] == "ALLOW"
    assert permission["allowed_modes"] == '["paper"]'


def test_publish_risk_permission_allows_live_ready_with_healthy_inputs(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)

    result = publish_risk_permission(lake, strategy="v5", version="5.0.0")

    assert result.permission == "ALLOW"
    assert result.reasons == ["all_required_alpha_gates_live_ready"]
    permission = read_parquet_dataset(lake / "gold" / "risk_permission").to_dicts()[0]
    assert permission["allowed_modes"] == '["paper"]'
    assert permission["allowed_live_modes"] == "[]"
    assert permission["max_gross_exposure"] == 0
    assert permission["max_single_weight"] == 0
    assert "quant_lab_live_command_not_allowed" in permission["live_block_reasons"]
    assert "v5_local_live_not_controlled_by_quant_lab" in permission["live_block_reasons"]


def test_publish_risk_permission_exposes_micro_canary_review_summary(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.QUARANTINE)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "trade_level_decision": "MICRO_CANARY_REVIEW",
                    "created_at": now,
                },
                {
                    "trade_level_decision": "MICRO_CANARY_REVIEW_BLOCKED_BY_OBSERVABILITY",
                    "created_at": now,
                },
            ]
        ),
        lake / "gold" / "trade_level_judgment",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "bucket_key": "SOL|f3|rank_1",
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "sample_count": 22,
                    "false_block_count": 11,
                    "loss_saved_count": 1,
                    "veto_net_value_bps": -2345.88,
                    "recommended_trade_level_decision": "MICRO_CANARY_REVIEW",
                    "created_at": now,
                }
            ]
        ),
        lake / "gold" / "opportunity_cost_by_bucket",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "bucket_key": "SOL|f3|rank_1",
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "sample_count": 22,
                    "false_block_count": 11,
                    "loss_saved_count": 1,
                    "veto_net_value_bps": -2345.88,
                    "policy_action": "MICRO_CANARY_REVIEW",
                    "policy_reason": "false_block_bucket_negative_veto_value_manual_review_only",
                    "created_at": now,
                }
            ]
        ),
        lake / "gold" / "trade_level_bucket_policy",
    )

    publish_risk_permission(lake, strategy="v5", version="5.0.0")

    permission = read_parquet_dataset(lake / "gold" / "risk_permission").to_dicts()[0]
    top_buckets = json.loads(permission["top_micro_canary_review_buckets"])
    advisory_modes = json.loads(permission["allowed_advisory_modes"])
    assert permission["micro_canary_review_count"] == 2
    assert permission["micro_canary_review_bucket_count"] == 1
    assert permission["blocked_by_observability_count"] == 1
    assert permission["reviewable_abort_count"] == 2
    assert permission["micro_canary_review_ready_count"] == 1
    assert permission["micro_canary_review_blocked_by_observability_count"] == 1
    assert permission["micro_canary_allow_candidate_count"] == 0
    assert permission["risk_block_bucket_count"] == 0
    assert permission["recommended_next_permission_mode"] == "MICRO_CANARY_REVIEW_ONLY"
    assert "micro_canary_review" in advisory_modes
    assert top_buckets[0]["bucket_key"] == "SOL|f3|rank_1"
    assert top_buckets[0]["policy_action"] == "MICRO_CANARY_REVIEW"
    assert permission["allowed_live_modes"] == "[]"
    assert permission["max_single_order_usdt"] == 0


def test_publish_risk_permission_data_health_warns_before_market_bar_stale(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.delenv("QUANT_LAB_MARKET_BAR_WARNING_DELAY_SECONDS", raising=False)
    monkeypatch.delenv("QUANT_LAB_MARKET_BAR_CRITICAL_DELAY_SECONDS", raising=False)
    _write_market_bar(lake, datetime.now(UTC) - timedelta(hours=3, minutes=5))

    health = lake_data_health(lake)

    assert health["status"] == "warning"
    assert health["is_critical"] is False
    assert health["reasons"] == ["market_bar_delayed"]
    assert health["latest_market_bar_close_ts"]


def test_publish_risk_permission_data_health_uses_market_bar_close_time(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.delenv("QUANT_LAB_MARKET_BAR_WARNING_DELAY_SECONDS", raising=False)
    monkeypatch.delenv("QUANT_LAB_MARKET_BAR_CRITICAL_DELAY_SECONDS", raising=False)
    _write_market_bar(lake, datetime.now(UTC) - timedelta(hours=2, minutes=5))

    health = lake_data_health(lake)

    assert health["status"] == "ok"
    assert health["freshness_seconds"] < 2 * 60 * 60
    assert health["latest_market_bar_close_ts"]


def test_publish_risk_permission_data_health_critical_after_market_bar_stale(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.delenv("QUANT_LAB_MARKET_BAR_WARNING_DELAY_SECONDS", raising=False)
    monkeypatch.delenv("QUANT_LAB_MARKET_BAR_CRITICAL_DELAY_SECONDS", raising=False)
    _write_market_bar(lake, datetime.now(UTC) - timedelta(hours=4, minutes=5))

    health = lake_data_health(lake)

    assert health["status"] == "critical"
    assert health["is_critical"] is True
    assert health["reasons"] == ["market_bar_stale"]


def test_publish_risk_permission_ignores_bootstrap_gate_when_research_gate_exists(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.LIVE_READY)
    _append_bootstrap_gate(lake)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)

    result = publish_risk_permission(lake, strategy="v5", version="5.0.0")

    assert result.permission == "ALLOW"
    permission = read_parquet_dataset(lake / "gold" / "risk_permission").to_dicts()[0]
    assert permission["gate_version"] == "default-v0.1"
    assert "bootstrap" not in permission["gate_version"]


def test_publish_risk_permission_aborts_on_gate_compliance_violation(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": datetime.now(UTC).date().isoformat(),
                    "status": "CRITICAL",
                    "violation_count": 1,
                }
            ]
        ),
        lake / "gold" / "v5_gate_compliance_daily",
    )

    result = publish_risk_permission(lake, strategy="v5", version="5.0.0")

    assert result.permission == "ABORT"
    assert "v5_gate_compliance_violation" in result.reasons


def test_publish_risk_permission_writes_idempotently(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.QUARANTINE)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)

    first = publish_risk_permission(lake, strategy="v5", version="5.0.0")
    second = publish_risk_permission(lake, strategy="v5", version="5.0.0")

    assert first.risk_permission_rows == second.risk_permission_rows == 1
    assert read_parquet_dataset(lake / "gold" / "risk_permission").height == 1
    meta_path = lake / "gold" / "risk_permission_api_dependency_meta" / "_snapshot_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["dataset"] == "risk_permission_api_dependency_meta"
    assert meta["row_count"] == 1
    assert meta["source_sha"]


def test_risk_permission_dependency_source_sha_covers_permission_fields():
    now = datetime.now(UTC).replace(microsecond=0)
    base = RiskPermission(
        strategy="v5",
        version="5.0.0",
        permission=RiskAction.ALLOW,
        allowed_modes=["paper"],
        allowed_live_modes=[],
        live_block_reasons=["quant_lab_live_command_not_allowed"],
        max_gross_exposure=0.0,
        max_single_weight=0.0,
        cost_model_version="costs-test",
        gate_version="default-v0.1",
        reasons=["same_reason"],
        created_at=now,
        as_of_ts=now,
        expires_at=now + timedelta(hours=1),
        telemetry_latest_ts=now,
        permission_status=RiskPermissionStatus.ACTIVE_ALLOW,
        enforceable=True,
        contract_version="risk_permission.v0.2",
    )

    base_sha = _dependency_source_sha(
        strategy="v5",
        version="5.0.0",
        permission=base,
        telemetry_latest_ts=now,
    )
    abort_sha = _dependency_source_sha(
        strategy="v5",
        version="5.0.0",
        permission=base.model_copy(update={"permission": RiskAction.ABORT}),
        telemetry_latest_ts=now,
    )
    live_mode_sha = _dependency_source_sha(
        strategy="v5",
        version="5.0.0",
        permission=base.model_copy(update={"allowed_live_modes": ["live_canary"]}),
        telemetry_latest_ts=now,
    )
    live_reason_sha = _dependency_source_sha(
        strategy="v5",
        version="5.0.0",
        permission=base.model_copy(
            update={"live_block_reasons": ["different_live_block_reason"]}
        ),
        telemetry_latest_ts=now,
    )

    assert len({base_sha, abort_sha, live_mode_sha, live_reason_sha}) == 4


def test_publish_risk_permission_marks_active_when_telemetry_within_threshold(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.QUARANTINE)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)
    telemetry_ts = datetime.now(UTC) - timedelta(minutes=5)
    _write_strategy_health(lake, telemetry_ts)

    publish_risk_permission(lake, strategy="v5", version="5.0.0")

    permission = read_parquet_dataset(lake / "gold" / "risk_permission").to_dicts()[0]
    assert permission["permission"] == "SELL_ONLY"
    assert permission["permission_status"] == "ACTIVE_SELL_ONLY"
    assert permission["telemetry_latest_ts"].startswith(telemetry_ts.isoformat()[:16])
    assert permission["contract_version"] == "risk_permission.v0.2"


def test_publish_risk_permission_prefers_bundle_manifest_over_health_day_boundary(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.QUARANTINE)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)
    manifest_ts = datetime.now(UTC) - timedelta(minutes=5)
    _write_v5_bundle_manifest(lake, manifest_ts)
    _write_strategy_health(lake, datetime.now(UTC) + timedelta(hours=24))

    publish_risk_permission(lake, strategy="v5", version="5.0.0")

    permission = read_parquet_dataset(lake / "gold" / "risk_permission").to_dicts()[0]
    assert permission["permission"] == "SELL_ONLY"
    assert permission["permission_status"] == "ACTIVE_SELL_ONLY"
    assert permission["telemetry_latest_ts"].startswith(manifest_ts.isoformat()[:16])


def test_publish_risk_permission_refreshes_expired_periodic_row(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.DEAD)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)
    expired_as_of = datetime.now(UTC) - timedelta(hours=2)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "schema_version": "v5.risk_permission.response.v0.2",
                    "strategy": "v5",
                    "version": "5.0.0",
                    "permission": "ABORT",
                    "allowed_modes": "[]",
                    "max_gross_exposure": 0.0,
                    "max_single_weight": 0.0,
                    "max_gross_exposure_usdt": 0.0,
                    "max_single_order_usdt": 0.0,
                    "cost_model_version": "costs-old",
                    "gate_version": "default-v0.1",
                    "reasons": '["old_abort"]',
                    "reason": "old_abort",
                    "created_at": expired_as_of.isoformat(),
                    "as_of_ts": expired_as_of.isoformat(),
                    "source_bundle_ts": None,
                    "expires_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
                    "telemetry_latest_ts": None,
                    "permission_freshness_sec": 0,
                    "contract_version": "risk_permission.v0.2",
                    "permission_status": "EXPIRED_ABORT",
                    "enforceable": False,
                    "risk_reason_codes": '["permission_expired"]',
                    "source": "test",
                    "fallback_level": "NONE",
                }
            ]
        ),
        lake / "gold" / "risk_permission",
    )

    publish_risk_permission(lake, strategy="v5", version="5.0.0")

    permission = read_parquet_dataset(lake / "gold" / "risk_permission").to_dicts()[0]
    assert permission["permission"] == "ABORT"
    assert permission["permission_status"] == "ACTIVE_ABORT"
    assert datetime.fromisoformat(permission["expires_at"]) > datetime.now(UTC)
    assert permission["enforceable"] is True


def test_publish_risk_permission_marks_stale_when_telemetry_is_newer_by_48h(tmp_path):
    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.DEAD)
    _write_fresh_market_bar(lake)
    _write_actual_cost(lake)
    _write_strategy_health(lake, datetime.now(UTC) + timedelta(hours=48))

    publish_risk_permission(lake, strategy="v5", version="5.0.0")

    permission = read_parquet_dataset(lake / "gold" / "risk_permission").to_dicts()[0]
    assert permission["permission"] == "ABORT"
    assert permission["permission_status"] == "STALE_ABORT"
    assert permission["permission_freshness_sec"] >= 47 * 60 * 60


def _write_gate(lake, status: GateStatus) -> None:
    decision = GateDecision(
        alpha_id="v5.required.alpha",
        version="v0.1",
        gate_version="default-v0.1",
        status=status,
        passed=status == GateStatus.LIVE_READY,
        reasons=[],
        metrics={"ic_tstat": 3.0},
        next_action="review",
        created_at=datetime.now(UTC),
    ).model_dump(mode="json")
    write_parquet_dataset(
        pl.DataFrame([decision | {"strategy": "v5"}]),
        lake / "gold" / "gate_decision",
    )


def _append_bootstrap_gate(lake) -> None:
    existing_rows = []
    for row in read_parquet_dataset(lake / "gold" / "gate_decision").to_dicts():
        normalized = dict(row)
        if isinstance(normalized.get("reasons"), list):
            normalized["reasons"] = "[]"
        if isinstance(normalized.get("metrics"), dict):
            normalized["metrics"] = '{"ic_tstat":3.0}'
        existing_rows.append(normalized)
    bootstrap = {
        "strategy": "v5",
        "alpha_id": "v5.core",
        "version": "bootstrap",
        "gate_version": "bootstrap.quarantine.v1",
        "status": "QUARANTINE",
        "passed": False,
        "reasons": '["bootstrap_gate"]',
        "metrics": '{"bootstrap":true}',
        "next_action": "generate_alpha_evidence_before_live",
        "created_at": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        "source": "bootstrap_gold_health",
        "fallback_level": "BOOTSTRAP_CONSERVATIVE",
    }
    write_parquet_dataset(
        pl.DataFrame([*existing_rows, bootstrap]),
        lake / "gold" / "gate_decision",
    )


def _write_fresh_market_bar(lake) -> None:
    _write_market_bar(lake, datetime.now(UTC) - timedelta(minutes=5))


def _write_market_bar(lake, ts: datetime) -> None:
    write_market_bars(
        lake,
        [
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": ts,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1.0,
                "quote_volume": 100.5,
                "source": "test",
                "ingest_ts": datetime.now(UTC),
            }
        ],
    )


def _write_actual_cost(lake) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": datetime.now(UTC).date().isoformat(),
                    "symbol": "BTC-USDT",
                    "regime": "realized",
                    "event_type": "actual_fill",
                    "notional_bucket": "all",
                    "sample_count": 30,
                    "fee_bps_p50": 1.0,
                    "fee_bps_p75": 1.0,
                    "fee_bps_p90": 1.0,
                    "slippage_bps_p50": 0.0,
                    "slippage_bps_p75": 0.0,
                    "slippage_bps_p90": 0.0,
                    "spread_bps_p50": 1.0,
                    "spread_bps_p75": 1.0,
                    "spread_bps_p90": 1.0,
                    "total_cost_bps_p50": 2.0,
                    "total_cost_bps_p75": 2.0,
                    "total_cost_bps_p90": 2.0,
                    "fallback_level": "NONE",
                    "source": "actual_okx_fills_and_bills",
                    "cost_model_version": "costs-test",
                    "created_at": datetime.now(UTC).isoformat(),
                }
            ]
        ),
        lake / "gold" / "cost_bucket_daily",
    )


def _write_strategy_health(lake, latest_bundle_ts: datetime) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": latest_bundle_ts.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": latest_bundle_ts,
                }
            ]
        ),
        lake / "gold" / "strategy_health_daily",
    )


def _write_v5_bundle_manifest(lake, bundle_ts: datetime) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": "sha-fixture",
                    "bundle_name": "v5_live_followup_bundle_fixture.tar.gz",
                    "bundle_ts": bundle_ts,
                    "ingest_ts": datetime.now(UTC),
                    "schema_version": "test",
                }
            ]
        ),
        lake / "bronze" / "strategy_telemetry" / "v5" / "bundle_manifest",
    )
