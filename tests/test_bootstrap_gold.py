import json
from datetime import UTC, datetime, timedelta

import polars as pl
from fastapi.testclient import TestClient

from quant_lab.api.main import app
from quant_lab.contracts.models import GateStatus, RiskAction
from quant_lab.costs.model import DEFAULT_FALLBACK_COST_BPS
from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.research.bootstrap_gold import (
    BOOTSTRAP_GATE_VERSION,
    BOOTSTRAP_WARNING,
    bootstrap_gold_health,
)
from quant_lab.web import readers


def test_bootstrap_gold_health_empty_lake_is_conservative(tmp_path):
    lake = tmp_path / "lake"

    result = bootstrap_gold_health(lake, strategy="v5", version="bootstrap", day="auto")

    assert result.cost_bootstrapped is True
    assert result.gate_bootstrapped is True
    assert result.risk_bootstrapped is True
    assert result.cost_bucket_daily_rows == 1
    assert result.gate_decision_rows == 1
    assert result.risk_permission_rows == 1

    cost = read_parquet_dataset(lake / "gold/cost_bucket_daily").to_dicts()[0]
    assert cost["symbol"] == "GLOBAL"
    assert cost["source"] == "global_default"
    assert cost["fallback_level"] == "GLOBAL_DEFAULT"
    assert cost["sample_count"] == 0

    gate = read_parquet_dataset(lake / "gold/gate_decision").to_dicts()[0]
    assert gate["strategy"] == "v5"
    assert gate["alpha_id"] == "v5.core"
    assert gate["status"] == GateStatus.QUARANTINE.value
    assert gate["status"] != GateStatus.LIVE_READY.value
    assert gate["passed"] is False
    assert gate["gate_version"] == BOOTSTRAP_GATE_VERSION
    assert json.loads(gate["reasons"]) == [
        "bootstrap_gate",
        "missing_alpha_evidence",
        "not_ready_for_live",
    ]

    permission = read_parquet_dataset(lake / "gold/risk_permission").to_dicts()[0]
    assert permission["permission"] == RiskAction.SELL_ONLY.value
    assert permission["permission"] != RiskAction.ALLOW.value
    assert "market_bar_missing" in json.loads(permission["reasons"])
    assert "cost_health_missing" in json.loads(permission["reasons"])


def test_bootstrap_gold_health_uses_latest_market_bar_day(tmp_path):
    lake = tmp_path / "lake"
    latest = datetime.now(UTC) - timedelta(minutes=10)
    write_market_bars(
        lake,
        [
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": latest,
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 12.0,
                "quote_volume": 1212.0,
                "source": "test",
                "ingest_ts": datetime.now(UTC),
            }
        ],
    )

    result = bootstrap_gold_health(lake, strategy="v5", version="bootstrap", day="auto")

    cost = read_parquet_dataset(lake / "gold/cost_bucket_daily").to_dicts()[0]
    assert result.day == latest.date().isoformat()
    assert cost["day"] == latest.date().isoformat()
    assert cost["symbol"] == "BTC-USDT"
    assert cost["source"] == "global_default"
    assert cost["sample_count"] == 0


def test_bootstrap_gold_health_is_idempotent(tmp_path):
    lake = tmp_path / "lake"

    first = bootstrap_gold_health(lake, strategy="v5", version="bootstrap", day="auto")
    second = bootstrap_gold_health(lake, strategy="v5", version="bootstrap", day="auto")

    assert first.cost_bucket_daily_rows == second.cost_bucket_daily_rows == 1
    assert first.gate_decision_rows == second.gate_decision_rows == 1
    assert first.risk_permission_rows == second.risk_permission_rows == 1
    assert read_parquet_dataset(lake / "gold/cost_bucket_daily").height == 1
    assert read_parquet_dataset(lake / "gold/gate_decision").height == 1
    assert read_parquet_dataset(lake / "gold/risk_permission").height == 1


def test_bootstrap_gold_health_feeds_cost_and_risk_api(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    bootstrap_gold_health(lake, strategy="v5", version="bootstrap", day="auto")
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    client = TestClient(app)

    risk_response = client.get(
        "/v1/risk/live-permission",
        params={"strategy": "v5", "version": "bootstrap"},
    )
    cost_response = client.get(
        "/v1/costs/estimate",
        params={
            "symbol": "BTC-USDT",
            "regime": "normal",
            "notional_usdt": 5_000,
            "quantile": "p75",
        },
    )

    assert risk_response.status_code == 200
    assert risk_response.json()["permission"] == RiskAction.SELL_ONLY.value
    assert "required_alpha_gate_quarantine" in risk_response.json()["reasons"]
    assert cost_response.status_code == 200
    assert cost_response.json()["source"] == "global_default"
    assert cost_response.json()["fallback_level"] == "GLOBAL_DEFAULT"
    assert cost_response.json()["total_cost_bps"] == DEFAULT_FALLBACK_COST_BPS


def test_web_diagnostics_show_bootstrap_warning_without_gold_missing(tmp_path):
    lake = tmp_path / "lake"
    bootstrap_gold_health(lake, strategy="v5", version="bootstrap", day="auto")

    diagnostics = readers.lake_diagnostics(lake)
    warnings = "\n".join(diagnostics["warnings"])

    assert "gold/cost_bucket_daily 数据集缺失或为空" not in warnings
    assert "gold/gate_decision 数据集缺失或为空" not in warnings
    assert "gold/risk_permission 数据集缺失或为空" not in warnings
    assert BOOTSTRAP_WARNING in warnings


def test_web_diagnostics_do_not_warn_when_latest_gate_and_risk_are_research_rows(tmp_path):
    lake = tmp_path / "lake"
    older = datetime(2026, 5, 11, 12, tzinfo=UTC)
    newer = datetime(2026, 5, 11, 20, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "alpha_id": "v5.core",
                    "version": "bootstrap",
                    "gate_version": BOOTSTRAP_GATE_VERSION,
                    "status": "QUARANTINE",
                    "passed": False,
                    "reasons": '["bootstrap_gate"]',
                    "metrics": '{"bootstrap":true}',
                    "next_action": "generate_alpha_evidence_before_live",
                    "created_at": older,
                    "source": "bootstrap_gold_health",
                    "fallback_level": "BOOTSTRAP_CONSERVATIVE",
                },
                {
                    "strategy": "v5",
                    "alpha_id": "v5.core.momentum",
                    "version": "v0.1",
                    "gate_version": "default-v0.1",
                    "status": "DEAD",
                    "passed": False,
                    "reasons": '["insufficient_coverage"]',
                    "metrics": '{"coverage":0.75}',
                    "next_action": "retire_alpha_or_research_new_hypothesis",
                    "created_at": newer,
                    "source": "research.alpha_evidence.v0.1",
                    "fallback_level": "NONE",
                },
            ]
        ),
        lake / "gold/gate_decision",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "bootstrap",
                    "permission": "SELL_ONLY",
                    "allowed_modes": '["sell_only"]',
                    "max_gross_exposure": 0.0,
                    "max_single_weight": 0.0,
                    "cost_model_version": "bootstrap.cost.v1",
                    "gate_version": BOOTSTRAP_GATE_VERSION,
                    "reasons": '["required_alpha_gate_quarantine"]',
                    "created_at": older,
                    "source": "bootstrap_gold_health",
                    "fallback_level": "BOOTSTRAP_CONSERVATIVE",
                },
                {
                    "strategy": "v5",
                    "version": "5.0.0",
                    "permission": "ABORT",
                    "allowed_modes": "[]",
                    "max_gross_exposure": 0.0,
                    "max_single_weight": 0.0,
                    "cost_model_version": "cost_bucket_daily:2026-05-11",
                    "gate_version": "default-v0.1",
                    "reasons": '["required_alpha_gate_dead"]',
                    "created_at": newer,
                    "source": "research.risk_permission.v0.1",
                    "fallback_level": "NONE",
                },
            ]
        ),
        lake / "gold/risk_permission",
    )

    warnings = "\n".join(readers.lake_diagnostics(lake)["warnings"])

    assert f"gold/gate_decision: {BOOTSTRAP_WARNING}" not in warnings
    assert f"gold/risk_permission: {BOOTSTRAP_WARNING}" not in warnings
