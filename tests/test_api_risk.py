from datetime import UTC, datetime, timedelta

import polars as pl
from fastapi.testclient import TestClient

from quant_lab.api.main import app
from quant_lab.contracts.models import GateDecision, GateStatus
from quant_lab.data.lake import write_market_bars, write_parquet_dataset


def test_live_permission_api_aborts_without_gate_decision(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_fresh_market_bar(lake)
    _write_cost_bucket(lake)

    response = TestClient(app).get(
        "/v1/risk/live-permission",
        params={"strategy": "v5", "version": "5.0.0"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["permission"] == "ABORT"
    assert payload["reasons"] == ["no_required_gate_decisions"]


def test_live_permission_api_aborts_on_stale_market_data(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_cost_bucket(lake)
    _write_stale_market_bar(lake)

    payload = _get_permission("v5")

    assert payload["permission"] == "ABORT"
    assert payload["reasons"] == ["market_bar_stale"]


def test_live_permission_api_sell_only_when_cost_missing(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_fresh_market_bar(lake)

    payload = _get_permission("v5")

    assert payload["permission"] == "SELL_ONLY"
    assert payload["allowed_modes"] == []
    assert payload["reasons"] == ["cost_health_missing"]
    assert payload["permission_status"] == "NO_FRESH_PERMISSION"
    assert payload["enforceable"] is False


def test_lake_data_health_uses_lazy_scan_not_full_market_bar_read(tmp_path, monkeypatch):
    from quant_lab.api.main import _lake_data_health

    lake = tmp_path / "lake"
    _write_fresh_market_bar(lake)

    def fail_full_read(*args, **kwargs):
        raise AssertionError("data health should not fully read market_bar")

    monkeypatch.setattr("quant_lab.api.main.read_parquet_dataset", fail_full_read)

    health = _lake_data_health(lake)

    assert health["status"] == "ok"
    assert health["latest_market_bar_ts"]


def test_lake_data_health_prefers_market_bar_health_metadata(tmp_path, monkeypatch):
    from quant_lab.api.main import _lake_data_health
    from quant_lab.api.main import read_parquet_lazy as real_read_parquet_lazy

    lake = tmp_path / "lake"
    _write_fresh_market_bar(lake)
    scanned_paths: list[str] = []

    def track_lazy_read(path):
        scanned_paths.append(str(path))
        return real_read_parquet_lazy(path)

    monkeypatch.setattr("quant_lab.api.main.read_parquet_lazy", track_lazy_read)

    health = _lake_data_health(lake)

    assert health["status"] == "ok"
    assert any(path.endswith("market_bar_health") for path in scanned_paths)
    assert not any(path.endswith("market_bar") for path in scanned_paths)


def test_lake_cost_health_uses_lazy_cost_bucket_fallback(tmp_path, monkeypatch):
    from quant_lab.api.main import _lake_cost_health

    lake = tmp_path / "lake"
    _write_cost_bucket(lake)

    def fail_full_read(*args, **kwargs):
        raise AssertionError("cost health fallback should lazy-read cost_bucket_daily")

    monkeypatch.setattr("quant_lab.api.main.read_parquet_dataset", fail_full_read)

    health = _lake_cost_health(lake)

    assert health["status"] == "ok"
    assert health["cost_model_version"].startswith("cost_bucket_daily:")


def test_risk_permission_selection_uses_lazy_strategy_version_filter(tmp_path, monkeypatch):
    from quant_lab.api.main import _select_published_risk_permission

    lake = tmp_path / "lake"
    as_of_ts = datetime.now(UTC)
    _write_risk_permissions(
        lake,
        [
            _risk_row(
                strategy="v7",
                version="5.0.0",
                permission="ALLOW",
                reasons='["wrong_strategy"]',
                as_of_ts=as_of_ts,
                permission_status="ACTIVE_ALLOW",
            ),
            _risk_row(
                strategy="v5",
                version="5.0.0",
                permission="ABORT",
                reasons='["selected_abort"]',
                as_of_ts=as_of_ts,
                permission_status="ACTIVE_ABORT",
            ),
        ],
    )

    def fail_full_read(*args, **kwargs):
        raise AssertionError("risk permission selection should lazy-filter rows")

    monkeypatch.setattr("quant_lab.api.main.read_parquet_dataset", fail_full_read)

    selection = _select_published_risk_permission(
        lake,
        strategy="v5",
        version="5.0.0",
        telemetry_latest_ts=None,
    )

    assert selection["active"] is not None
    assert selection["active"].reasons == ["selected_abort"]


def test_gate_decision_loading_uses_lazy_strategy_filter(tmp_path, monkeypatch):
    from quant_lab.api.main import _load_gate_decisions

    lake = tmp_path / "lake"
    _write_gate(lake, GateStatus.LIVE_READY)

    def fail_full_read(*args, **kwargs):
        raise AssertionError("gate decision loading should lazy-filter rows")

    monkeypatch.setattr("quant_lab.api.main.read_parquet_dataset", fail_full_read)

    decisions = _load_gate_decisions(lake, strategy="v5")

    assert len(decisions) == 1
    assert decisions[0].status == GateStatus.LIVE_READY


def test_strategy_telemetry_reasons_use_lazy_latest_rows(tmp_path, monkeypatch):
    from quant_lab.api.main import _strategy_telemetry_reasons

    lake = tmp_path / "lake"
    now = datetime.now(UTC)
    _write_strategy_health(lake, now)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": now.date().isoformat(),
                    "status": "OK",
                    "violation_count": 1,
                }
            ]
        ),
        lake / "gold/v5_gate_compliance_daily",
    )

    def fail_full_read(*args, **kwargs):
        raise AssertionError("telemetry reasons should lazy-read latest rows")

    monkeypatch.setattr("quant_lab.api.main.read_parquet_dataset", fail_full_read)

    assert _strategy_telemetry_reasons(lake, strategy="v5") == ["v5_gate_compliance_violation"]


def test_latest_strategy_telemetry_ts_uses_lazy_aggregation(tmp_path, monkeypatch):
    from quant_lab.risk.publish import latest_strategy_telemetry_ts

    lake = tmp_path / "lake"
    latest_ts = datetime.now(UTC).replace(microsecond=0)
    _write_strategy_health(lake, latest_ts)

    def fail_full_read(*args, **kwargs):
        raise AssertionError("latest strategy telemetry should lazy-aggregate")

    monkeypatch.setattr("quant_lab.risk.publish.read_parquet_dataset", fail_full_read)

    assert latest_strategy_telemetry_ts(lake, "v5") == latest_ts


def test_live_permission_api_requires_published_permission_even_when_inputs_are_healthy(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_cost_bucket(lake)
    _write_fresh_market_bar(lake)

    payload = _get_permission("v5")

    assert payload["permission"] == "ALLOW"
    assert payload["allowed_modes"] == []
    assert payload["reasons"] == ["all_required_alpha_gates_live_ready"]
    assert payload["cost_model_version"].startswith("cost_bucket_daily:")
    assert payload["gate_version"] == "default-v0.1"
    assert payload["permission_status"] == "NO_FRESH_PERMISSION"
    assert payload["enforceable"] is False
    assert "no_fresh_published_permission" in payload["risk_reason_codes"]
    assert payload["reason"] == "no_fresh_published_permission"


def test_live_permission_api_aborts_on_v5_gate_compliance_violation(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_cost_bucket(lake)
    _write_fresh_market_bar(lake)
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
        lake / "gold/v5_gate_compliance_daily",
    )

    payload = _get_permission("v5")

    assert payload["permission"] == "ABORT"
    assert "v5_gate_compliance_violation" in payload["reasons"]


def test_live_permission_api_allows_advisory_modes_when_core_alpha_dead(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.DEAD, alpha_id="v5.core.momentum")
    _write_cost_bucket(lake)
    _write_fresh_market_bar(lake)
    _write_alpha_discovery_board(lake, decision="PAPER_READY")

    payload = _get_permission("v5")

    assert payload["permission"] == "ABORT"
    assert payload["permission"] != "ALLOW"
    assert payload["core_alpha_gate_status"] == "DEAD"
    assert payload["core_alpha_dead"] is True
    assert payload["baseline_status"] == "DEAD"
    assert payload["baseline_role"] == "research_baseline"
    assert payload["baseline_not_live_eligible"] is True
    assert payload["baseline_not_global_strategy_gate"] is True
    assert payload["system_safety_status"] == "SAFE_FOR_ADVISORY"
    assert payload["strategy_opportunities_available"] is True
    assert payload["allowed_advisory_modes"] == ["shadow", "paper"]
    assert payload["allowed_live_modes"] == []
    assert "baseline_not_global_strategy_gate" in payload["live_block_reasons"]
    assert "quant_lab_live_command_not_allowed" in payload["live_block_reasons"]
    assert "v5_local_live_not_controlled_by_quant_lab" in payload["live_block_reasons"]


def test_live_permission_advisory_modes_use_latest_alpha_board_day(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.DEAD, alpha_id="v5.core.momentum")
    _write_cost_bucket(lake)
    _write_fresh_market_bar(lake)
    created_at = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "as_of_date": (created_at.date() - timedelta(days=1)).isoformat(),
                    "strategy_candidate": "v5.old_candidate",
                    "symbol": "SOL-USDT",
                    "horizon_hours": 24,
                    "sample_count": 72,
                    "complete_sample_count": 72,
                    "avg_net_bps": 42.0,
                    "p25_net_bps": 3.0,
                    "win_rate": 0.72,
                    "cost_source_mix": '[{"cost_source":"public_spread_proxy","count":72}]',
                    "decision": "PAPER_READY",
                    "created_at": created_at - timedelta(days=1),
                },
                {
                    "strategy": "v5",
                    "as_of_date": created_at.date().isoformat(),
                    "strategy_candidate": "v5.current_candidate",
                    "symbol": "SOL-USDT",
                    "horizon_hours": 24,
                    "sample_count": 72,
                    "complete_sample_count": 72,
                    "avg_net_bps": -30.0,
                    "p25_net_bps": -80.0,
                    "win_rate": 0.30,
                    "cost_source_mix": '[{"cost_source":"public_spread_proxy","count":72}]',
                    "decision": "KILL",
                    "created_at": created_at,
                },
            ]
        ),
        lake / "gold/alpha_discovery_board",
    )

    payload = _get_permission("v5")

    assert payload["permission"] == "ABORT"
    assert payload["strategy_opportunities_available"] is False
    assert payload["allowed_advisory_modes"] == []


def test_live_permission_api_does_not_use_published_allow_when_core_alpha_dead(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.setenv("QUANT_LAB_RISK_PERMISSION_TTL_SECONDS", "3600")
    _write_gate(lake, GateStatus.DEAD, alpha_id="v5.core.momentum")
    _write_cost_bucket(lake)
    _write_fresh_market_bar(lake)
    _write_alpha_discovery_board(lake, decision="PAPER_READY")
    as_of_ts = datetime.now(UTC)
    _write_risk_permissions(
        lake,
        [
            _risk_row(
                strategy="v5",
                version="v1",
                permission="ALLOW",
                reasons='["published_allow_should_not_override_dead_core_alpha"]',
                as_of_ts=as_of_ts,
                expires_at=as_of_ts + timedelta(hours=1),
                source_bundle_ts=as_of_ts,
                permission_status="ACTIVE_ALLOW",
            )
            | {
                "allowed_modes": '["paper","live_canary"]',
                "max_gross_exposure": 0.25,
                "max_single_weight": 0.05,
            }
        ],
    )

    payload = _get_permission("v5")

    assert payload["permission"] == "ABORT"
    assert payload["permission"] != "ALLOW"
    assert payload["core_alpha_dead"] is True
    assert payload["strategy_opportunities_available"] is True
    assert payload["allowed_advisory_modes"] == ["shadow", "paper"]
    assert payload["allowed_live_modes"] == []
    assert payload["system_safety_status"] == "SAFE_FOR_ADVISORY"
    assert "baseline_not_global_strategy_gate" in payload["live_block_reasons"]
    assert "quant_lab_live_command_not_allowed" in payload["live_block_reasons"]
    assert "v5_local_live_not_controlled_by_quant_lab" in payload["live_block_reasons"]

    detail = (
        TestClient(app)
        .get(
            "/v1/risk/live-permission-detail",
            params={"strategy": "v5", "version": "v1"},
        )
        .json()
    )
    assert detail["permission_source"] == "recomputed"
    assert detail["permission"]["permission"] == "ABORT"


def test_live_permission_api_reads_published_risk_permission(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_cost_bucket(lake)
    _write_fresh_market_bar(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "5.0.0",
                    "permission": "SELL_ONLY",
                    "allowed_modes": '["sell_only"]',
                    "max_gross_exposure": 0.0,
                    "max_single_weight": 0.0,
                    "cost_model_version": "costs-test",
                    "gate_version": "default-v0.1",
                    "reasons": '["published_permission"]',
                    "created_at": datetime.now(UTC).isoformat(),
                    "source": "test",
                    "fallback_level": "NONE",
                }
            ]
        ),
        lake / "gold/risk_permission",
    )

    response = TestClient(app).get(
        "/v1/risk/live-permission",
        params={"strategy": "v5", "version": "5.0.0"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["permission"] == "SELL_ONLY"
    assert payload["reasons"] == ["published_permission"]
    assert "permission_source" not in payload
    assert payload["permission_status"] == "ACTIVE_SELL_ONLY"
    assert payload["enforceable"] is True
    assert payload["max_gross_exposure_usdt"] == 0.0
    assert payload["max_single_order_usdt"] == 0.0
    assert payload["reason"] == "published_permission"

    detail = TestClient(app).get(
        "/v1/risk/live-permission-detail",
        params={"strategy": "v5", "version": "5.0.0"},
    )
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["permission"]["permission"] == "SELL_ONLY"
    assert detail_payload["permission_source"] == "published_cache"
    assert detail_payload["published_permission_stale"] is False
    assert detail_payload["permission_health"]["latest_permission_status"] == "ACTIVE_SELL_ONLY"
    assert detail_payload["permission_health"]["expires_in_sec"] is not None
    assert detail_payload["permission_health"]["refresh_lag_sec"] is not None
    assert detail_payload["permission_health"]["stale_permission_consecutive_count"] == 0
    assert detail_payload["permission_freshness_seconds"] >= 0
    assert detail_payload["data_health"]["status"] == "ok"
    assert detail_payload["cost_health"]["status"] == "ok"
    assert detail_payload["gate_summary"]["status_counts"] == {"LIVE_READY": 1}


def test_live_permission_api_recomputes_stale_published_allow(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.setenv("QUANT_LAB_RISK_PERMISSION_TTL_SECONDS", "300")
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_cost_bucket(lake)
    _write_stale_market_bar(lake)
    as_of_ts = datetime.now(UTC) - timedelta(hours=2)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "5.0.0",
                    "permission": "ALLOW",
                    "allowed_modes": '["paper","live_canary"]',
                    "max_gross_exposure": 0.25,
                    "max_single_weight": 0.05,
                    "cost_model_version": "costs-test",
                    "gate_version": "default-v0.1",
                    "reasons": '["old_allow"]',
                    "created_at": as_of_ts.isoformat(),
                    "as_of_ts": as_of_ts.isoformat(),
                    "source_bundle_ts": datetime.now(UTC).isoformat(),
                    "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                    "source": "test",
                    "fallback_level": "NONE",
                }
            ]
        ),
        lake / "gold/risk_permission",
    )

    payload = _get_permission("v5")

    assert payload["permission"] == "ABORT"
    assert "market_bar_stale" in payload["reasons"]

    detail = (
        TestClient(app)
        .get(
            "/v1/risk/live-permission-detail",
            params={"strategy": "v5", "version": "5.0.0"},
        )
        .json()
    )
    assert detail["permission_source"] == "recomputed_stale_context"
    assert detail["published_permission_stale"] is True
    assert detail["permission"]["permission"] == "ABORT"
    assert detail["permission"]["permission_status"] == "STALE_ABORT"
    assert detail["permission"]["enforceable"] is False


def test_live_permission_detail_overrides_fresh_allow_when_recomputed_is_more_conservative(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.setenv("QUANT_LAB_RISK_PERMISSION_TTL_SECONDS", "300")
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_cost_bucket(lake)
    _write_stale_market_bar(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "5.0.0",
                    "permission": "ALLOW",
                    "allowed_modes": '["paper","live_canary"]',
                    "max_gross_exposure": 0.25,
                    "max_single_weight": 0.05,
                    "cost_model_version": "costs-test",
                    "gate_version": "default-v0.1",
                    "reasons": '["fresh_allow"]',
                    "created_at": datetime.now(UTC).isoformat(),
                    "source": "test",
                    "fallback_level": "NONE",
                }
            ]
        ),
        lake / "gold/risk_permission",
    )

    detail = (
        TestClient(app)
        .get(
            "/v1/risk/live-permission-detail",
            params={"strategy": "v5", "version": "5.0.0"},
        )
        .json()
    )

    assert detail["permission_source"] == "recomputed"
    assert detail["published_permission_stale"] is False
    assert detail["permission"]["permission"] == "ABORT"


def test_live_permission_detail_recomputes_permission_stale_vs_v5_telemetry(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.setenv("QUANT_LAB_RISK_PERMISSION_TTL_SECONDS", "999999")
    _write_gate(lake, GateStatus.DEAD)
    _write_cost_bucket(lake)
    _write_fresh_market_bar(lake)
    as_of_ts = datetime.now(UTC) - timedelta(hours=48)
    telemetry_ts = datetime.now(UTC)
    _write_strategy_health(lake, telemetry_ts)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "5.0.0",
                    "permission": "ABORT",
                    "allowed_modes": "[]",
                    "max_gross_exposure": 0.0,
                    "max_single_weight": 0.0,
                    "cost_model_version": "costs-test",
                    "gate_version": "default-v0.1",
                    "reasons": '["old_abort"]',
                    "created_at": as_of_ts.isoformat(),
                    "as_of_ts": as_of_ts.isoformat(),
                    "permission_status": "STALE_ABORT",
                    "source": "test",
                    "fallback_level": "NONE",
                }
            ]
        ),
        lake / "gold/risk_permission",
    )

    detail = (
        TestClient(app)
        .get(
            "/v1/risk/live-permission-detail",
            params={"strategy": "v5", "version": "5.0.0"},
        )
        .json()
    )

    assert detail["published_permission_stale"] is True
    assert detail["permission_source"] == "published_stale"
    assert detail["permission"]["permission"] == "ABORT"
    assert detail["permission"]["permission_status"] == "STALE_ABORT"
    assert detail["permission"]["enforceable"] is False
    assert detail["permission"]["as_of_ts"] == as_of_ts.isoformat().replace("+00:00", "Z")


def test_live_permission_api_returns_latest_active_matching_strategy_version(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.setenv("QUANT_LAB_RISK_PERMISSION_TTL_SECONDS", "999999")
    old_as_of = datetime.now(UTC) - timedelta(days=2)
    new_as_of = datetime.now(UTC) - timedelta(minutes=5)
    _write_risk_permissions(
        lake,
        [
            _risk_row(
                strategy="v5",
                version="5.0.0",
                permission="ABORT",
                reasons='["old_abort"]',
                as_of_ts=old_as_of,
                permission_status="ACTIVE_ABORT",
            ),
            _risk_row(
                strategy="v5",
                version="5.0.0",
                permission="ABORT",
                reasons='["new_abort"]',
                as_of_ts=new_as_of,
                permission_status="ACTIVE_ABORT",
            ),
        ],
    )

    response = TestClient(app).get(
        "/v1/risk/live-permission",
        params={"strategy": "v5", "version": "5.0.0"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reasons"] == ["new_abort"]
    assert payload["as_of_ts"] == new_as_of.isoformat().replace("+00:00", "Z")
    assert payload["permission_status"] == "ACTIVE_ABORT"
    assert payload["enforceable"] is True


def test_live_permission_api_breaks_ties_by_latest_source_bundle_ts(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.setenv("QUANT_LAB_RISK_PERMISSION_TTL_SECONDS", "999999")
    as_of = datetime.now(UTC) - timedelta(minutes=5)
    old_bundle = as_of - timedelta(minutes=2)
    new_bundle = as_of - timedelta(minutes=1)
    _write_risk_permissions(
        lake,
        [
            _risk_row(
                strategy="v5",
                version="5.0.0",
                permission="ABORT",
                reasons='["old_bundle_abort"]',
                as_of_ts=as_of,
                source_bundle_ts=old_bundle,
                permission_status="ACTIVE_ABORT",
            ),
            _risk_row(
                strategy="v5",
                version="5.0.0",
                permission="ABORT",
                reasons='["new_bundle_abort"]',
                as_of_ts=as_of,
                source_bundle_ts=new_bundle,
                permission_status="ACTIVE_ABORT",
            ),
        ],
    )

    payload = (
        TestClient(app)
        .get(
            "/v1/risk/live-permission",
            params={"strategy": "v5", "version": "5.0.0"},
        )
        .json()
    )

    assert payload["reasons"] == ["new_bundle_abort"]
    assert payload["source_bundle_ts"] == new_bundle.isoformat().replace("+00:00", "Z")


def test_live_permission_api_downgrades_expired_active_permission(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    as_of = datetime.now(UTC) - timedelta(minutes=10)
    _write_risk_permissions(
        lake,
        [
            _risk_row(
                strategy="v5",
                version="5.0.0",
                permission="ABORT",
                reasons='["expired_abort"]',
                as_of_ts=as_of,
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
                permission_status="ACTIVE_ABORT",
            ),
        ],
    )

    payload = (
        TestClient(app)
        .get(
            "/v1/risk/live-permission",
            params={"strategy": "v5", "version": "5.0.0"},
        )
        .json()
    )

    assert payload["permission"] == "ABORT"
    assert payload["permission_status"] == "NO_FRESH_PERMISSION"
    assert payload["enforceable"] is False
    assert "no_fresh_published_permission" in payload["risk_reason_codes"]


def test_live_permission_api_no_fresh_permission_has_empty_allowed_modes(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    _write_gate(lake, GateStatus.LIVE_READY)
    _write_cost_bucket(lake)
    _write_fresh_market_bar(lake)

    payload = (
        TestClient(app)
        .get(
            "/v1/risk/live-permission",
            params={"strategy": "v5", "version": "5.0.0"},
        )
        .json()
    )

    assert payload["permission_status"] == "NO_FRESH_PERMISSION"
    assert payload["enforceable"] is False
    assert payload["allowed_modes"] == []
    assert payload["max_gross_exposure_usdt"] == 0.0
    assert payload["max_single_order_usdt"] == 0.0


def test_live_permission_detail_flags_response_older_than_gold_latest(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.setenv("QUANT_LAB_RISK_PERMISSION_TTL_SECONDS", "999999")
    active_as_of = datetime.now(UTC) - timedelta(minutes=20)
    stale_as_of = datetime.now(UTC) - timedelta(minutes=5)
    _write_strategy_health(lake, datetime.now(UTC))
    _write_risk_permissions(
        lake,
        [
            _risk_row(
                strategy="v5",
                version="5.0.0",
                permission="ABORT",
                reasons='["active_abort"]',
                as_of_ts=active_as_of,
                source_bundle_ts=active_as_of,
                permission_status="ACTIVE_ABORT",
            ),
            _risk_row(
                strategy="v5",
                version="5.0.0",
                permission="ABORT",
                reasons='["newer_stale_abort"]',
                as_of_ts=stale_as_of,
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
                source_bundle_ts=stale_as_of,
                permission_status="STALE_ABORT",
            ),
        ],
    )

    detail = (
        TestClient(app)
        .get(
            "/v1/risk/live-permission-detail",
            params={"strategy": "v5", "version": "5.0.0"},
        )
        .json()
    )

    assert detail["permission"]["as_of_ts"] == active_as_of.isoformat().replace("+00:00", "Z")
    assert detail["api_consistency"]["compare_gold_latest_vs_api_response"] == "FAIL"


def _get_permission(strategy: str) -> dict:
    response = TestClient(app).get(
        "/v1/risk/live-permission",
        params={"strategy": strategy, "version": "v1"},
    )
    assert response.status_code == 200
    return response.json()


def _write_gate(lake, status: GateStatus, alpha_id: str = "alpha-test") -> None:
    decision = GateDecision(
        alpha_id=alpha_id,
        version="v1",
        gate_version="default-v0.1",
        status=status,
        passed=status == GateStatus.LIVE_READY,
        reasons=[],
        metrics={"ic_tstat": 3.0},
        next_action="allow canary" if status == GateStatus.LIVE_READY else "review",
        created_at=datetime.now(UTC),
    ).model_dump(mode="json")
    write_parquet_dataset(
        pl.DataFrame([decision | {"strategy": "v5"}]),
        lake / "gold/gate_decision",
    )


def _write_alpha_discovery_board(lake, decision: str) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "as_of_date": datetime.now(UTC).date().isoformat(),
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "candidate_name": "v5.f4_volume_expansion_entry",
                    "source_type": "test",
                    "symbol": "SOL-USDT",
                    "regime_state": "trend",
                    "horizon_hours": 24,
                    "sample_count": 72,
                    "complete_sample_count": 72,
                    "avg_net_bps": 42.0,
                    "p25_net_bps": 3.0,
                    "win_rate": 0.72,
                    "cost_source_mix": '[{"cost_source":"public_spread_proxy","count":72}]',
                    "decision": decision,
                    "decision_reasons": '["paper_ready_thresholds_met"]',
                    "created_at": datetime.now(UTC),
                }
            ]
        ),
        lake / "gold/alpha_discovery_board",
    )


def _write_cost_bucket(lake) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": datetime.now(UTC).date().isoformat(),
                    "symbol": "BTC-USDT",
                    "regime": "normal",
                    "event_type": "trade",
                    "notional_bucket": "all",
                    "sample_count": 30,
                    "fee_bps_p50": 1.0,
                    "fee_bps_p75": 1.5,
                    "fee_bps_p90": 2.0,
                    "slippage_bps_p50": 1.0,
                    "slippage_bps_p75": 1.5,
                    "slippage_bps_p90": 2.0,
                    "spread_bps_p50": 0.5,
                    "spread_bps_p75": 0.75,
                    "spread_bps_p90": 1.0,
                    "total_cost_bps_p50": 2.5,
                    "total_cost_bps_p75": 3.75,
                    "total_cost_bps_p90": 5.0,
                    "fallback_level": "actual_okx_fills_and_bills",
                    "source": "actual_okx_fills_and_bills",
                }
            ]
        ),
        lake / "gold/cost_bucket_daily",
    )


def _write_fresh_market_bar(lake) -> None:
    _write_market_bar(lake, datetime.now(UTC) - timedelta(minutes=5))


def _write_stale_market_bar(lake) -> None:
    _write_market_bar(lake, datetime.now(UTC) - timedelta(days=2))


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
                "high": 110.0,
                "low": 90.0,
                "close": 105.0,
                "volume": 12.0,
                "quote_volume": 1260.0,
                "source": "test",
                "ingest_ts": datetime.now(UTC),
            }
        ],
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
        lake / "gold/strategy_health_daily",
    )


def _write_risk_permissions(lake, rows: list[dict]) -> None:
    write_parquet_dataset(pl.DataFrame(rows), lake / "gold/risk_permission")


def _risk_row(
    *,
    strategy: str,
    version: str,
    permission: str,
    reasons: str,
    as_of_ts: datetime,
    permission_status: str,
    expires_at: datetime | None = None,
    source_bundle_ts: datetime | None = None,
) -> dict:
    return {
        "strategy": strategy,
        "version": version,
        "permission": permission,
        "allowed_modes": "[]",
        "max_gross_exposure": 0.0,
        "max_single_weight": 0.0,
        "cost_model_version": "costs-test",
        "gate_version": "default-v0.1",
        "reasons": reasons,
        "created_at": as_of_ts.isoformat(),
        "as_of_ts": as_of_ts.isoformat(),
        "source_bundle_ts": source_bundle_ts.isoformat() if source_bundle_ts else None,
        "expires_at": expires_at.isoformat() if expires_at else None,
        "permission_status": permission_status,
        "contract_version": "risk_permission.v0.2",
        "source": "test",
        "fallback_level": "NONE",
    }
