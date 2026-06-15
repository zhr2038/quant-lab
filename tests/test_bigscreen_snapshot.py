from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from fastapi.testclient import TestClient

from quant_lab.api.main import create_app
from quant_lab.data.lake import write_parquet_dataset
from quant_lab.web.bigscreen import _exports_payload, bigscreen_snapshot, clear_bigscreen_cache


def test_bigscreen_snapshot_empty_lake_is_read_only_and_degraded(tmp_path):
    clear_bigscreen_cache()
    payload = bigscreen_snapshot(tmp_path / "lake")

    assert payload["mode"] == "read-only"
    assert payload["status"] in {"WARNING", "CRITICAL", "OK"}
    assert 0 <= payload["health_score"] <= 100
    assert isinstance(payload["actions"], list)
    assert len(payload["actions"]) <= 8
    assert payload["data_matrix"]["columns"] == [
        "market_bar",
        "ws",
        "spread",
        "trade",
        "cost",
        "evidence",
        "advisory",
    ]


def test_bigscreen_snapshot_redacts_secret_like_fields(tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "permission": "ALLOW",
                    "permission_status": "ALLOW",
                    "api_key": "should-not-leak",
                    "as_of_ts": "2026-06-04T00:00:00Z",
                }
            ]
        ),
        lake / "gold" / "risk_permission",
    )

    payload = bigscreen_snapshot(lake)
    rows = payload["consumers"]["permission_rows"]

    assert rows
    assert rows[0]["api_key"] == "[REDACTED]"
    assert "should-not-leak" not in str(payload)


def test_bigscreen_snapshot_exposes_factor_factory_results(tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    created_at = datetime(2026, 6, 6, 13, 0, tzinfo=UTC)
    exports = tmp_path / "exports"
    exports.mkdir()
    bridge_csv = "\n".join(
        [
            (
                "as_of_date,factor_id,factor_family,correlation_cluster_id,symbol,"
                "regime,horizon,horizon_hours,forward_sample_count,"
                "forward_cost_adjusted_score,bridge_candidate_id,"
                "eligible_for_alpha_factory,blocking_reasons,recommended_action,"
                "live_order_effect"
            ),
            (
                "2026-06-06,factor.momentum_zscore,momentum,cluster_001,"
                "SOL-USDT,TREND_UP,4h-8h,\"4,8\",122,108.0,"
                "v5.factor_bridge.factor.momentum_zscore,strategy_review_pending,"
                "\"[\"\"alpha_factory_strategy_review_required\"\"]\","
                "REVIEW_FOR_ALPHA_FACTORY_STRATEGY,none_read_only_research"
            ),
        ]
    )
    with zipfile.ZipFile(
        exports / "quant_lab_expert_pack_2026-06-06_20260606T130000+0000.zip",
        "w",
    ) as archive:
        archive.writestr("reports/factor_strategy_bridge_candidates.csv", bridge_csv)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-06-06",
                    "factor_id": "factor.momentum_zscore",
                    "factor_name": "Momentum zscore",
                    "factor_family": "momentum",
                    "factor_version": "v0.1",
                    "timeframe": "1H",
                    "best_horizon_bars": 24,
                    "tested_horizon_count": 4,
                    "best_score": 0.82,
                    "avg_score": 0.41,
                    "best_rank_ic_mean": 0.12,
                    "best_rank_ic_tstat": 3.1,
                    "best_long_short_mean_bps": 96.5,
                    "candidate_state": "PAPER_READY",
                    "recommended_action": "paper_review_only",
                    "promotion_block_reasons_json": "[]",
                    "manual_review_required": True,
                    "created_at": created_at,
                    "source": "factor_factory_v0.1",
                },
                {
                    "as_of_date": "2026-06-06",
                    "factor_id": "factor.noise",
                    "factor_name": "Noise",
                    "factor_family": "diagnostic",
                    "factor_version": "v0.1",
                    "timeframe": "1H",
                    "best_horizon_bars": 8,
                    "tested_horizon_count": 4,
                    "best_score": -0.2,
                    "avg_score": -0.3,
                    "best_rank_ic_mean": -0.02,
                    "best_rank_ic_tstat": -0.5,
                    "best_long_short_mean_bps": -12.0,
                    "candidate_state": "KILL",
                    "recommended_action": "suppress",
                    "promotion_block_reasons_json": '["negative_edge"]',
                    "manual_review_required": False,
                    "created_at": created_at,
                    "source": "factor_factory_v0.1",
                },
            ]
        ),
        lake / "gold" / "factor_candidate",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-06-06",
                    "factor_id": "factor.momentum_zscore",
                    "factor_name": "Momentum zscore",
                    "factor_family": "momentum",
                    "factor_version": "v0.1",
                    "timeframe": "1H",
                    "horizon_bars": 24,
                    "decision_delay_bars": 1,
                    "sample_count": 240,
                    "rank_ic_mean": 0.12,
                    "rank_ic_tstat": 3.1,
                    "long_short_mean_bps": 96.5,
                    "win_rate": 0.63,
                    "hit_rate": 0.61,
                    "turnover": 0.2,
                    "max_drawdown": 0.08,
                    "edge_cost_ratio": 2.1,
                    "cost_ratio": 0.3,
                    "period_count": 240,
                    "decision": "PAPER_READY",
                    "score": 0.82,
                    "reasons_json": "[]",
                    "warnings_json": "[]",
                    "start_ts": created_at,
                    "end_ts": created_at,
                    "created_at": created_at,
                    "source": "factor_factory_v0.1",
                }
            ]
        ),
        lake / "gold" / "factor_evidence",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-06-06",
                    "factor_id_left": "factor.momentum_zscore",
                    "factor_id_right": "factor.momentum_slope",
                    "factor_version": "v0.1",
                    "timeframe": "1H",
                    "sample_count": 240,
                    "correlation": 0.94,
                    "created_at": created_at,
                    "source": "factor_factory_v0.1",
                }
            ]
        ),
        lake / "gold" / "factor_correlation_daily",
    )

    payload = bigscreen_snapshot(lake)
    factor_factory = payload["strategy_flow"]["factor_factory"]

    assert factor_factory["live_order_effect"] == "none_read_only_research"
    assert factor_factory["candidate_count"] == 2
    assert factor_factory["paper_ready_count"] == 1
    assert factor_factory["paper_ready_candidates"][0]["factor_id"] == "factor.momentum_zscore"
    assert factor_factory["evidence_by_horizon"][0]["horizon_bars"] == 24
    assert factor_factory["high_correlation_pairs"][0]["correlation"] == 0.94
    assert factor_factory["family_leaderboard"]
    assert factor_factory["paper_review_queue"][0]["factor_id"] == "factor.momentum_zscore"
    assert factor_factory["composite_candidates"]
    assert "strategy_bridge_candidates" in factor_factory
    bridge = factor_factory["strategy_bridge_candidates"][0]
    assert bridge["factor_id"] == "factor.momentum_zscore"
    assert bridge["symbol"] == "SOL-USDT"
    assert bridge["regime"] == "TREND_UP"
    assert bridge["horizon"] == "4h-8h"


def test_bigscreen_snapshot_surfaces_legacy_web_anomalies(monkeypatch, tmp_path):
    clear_bigscreen_cache()

    def fake_data_health_summary(_lake_root):
        return {
            "latest_per_symbol": pl.DataFrame(),
            "missing_bars": pl.DataFrame(
                [
                    {
                        "symbol": "BTC-USDT",
                        "timeframe": "1H",
                        "missing_bars": 2,
                    }
                ]
            ),
            "duplicate_bar_count": 1,
            "unclosed_bar_count": 1,
            "schema_violations": ["market_bar missing close"],
            "schema_violation_count": 1,
            "stale_datasets": pl.DataFrame(
                [
                    {
                        "dataset": "market_regime_daily",
                        "status": "过期",
                        "severity": "WARNING",
                        "takeaway": "旧页面下游可能看到旧结果",
                        "next_action": "刷新 research 任务",
                    }
                ]
            ),
            "latest_market_bar_ts": datetime(2026, 6, 10, 1, tzinfo=UTC),
            "missing_bar_ratio": 0.02,
            "warnings": ["legacy warning"],
        }

    monkeypatch.setattr(
        "quant_lab.web.bigscreen.readers.data_health_summary",
        fake_data_health_summary,
    )

    payload = bigscreen_snapshot(tmp_path / "lake")
    legacy = payload["legacy_anomalies"]

    assert legacy["live_order_effect"] == "none_read_only_display"
    assert legacy["has_anomalies"] is True
    assert legacy["stale_dataset_count"] == 1
    assert legacy["schema_violation_count"] == 1
    assert legacy["unclosed_bar_count"] == 1
    assert legacy["duplicate_bar_count"] == 1
    assert legacy["missing_bar_count"] == 2
    assert {item["source"] for item in legacy["items"]} >= {
        "legacy_data_health.stale_datasets",
        "legacy_data_health.market_bar",
        "legacy_data_health.missing_bars",
    }
    assert any(
        action.get("source") == "legacy_data_health.stale_datasets"
        for action in payload["actions"]
    )


def test_bigscreen_snapshot_endpoints_return_payload(monkeypatch, tmp_path):
    clear_bigscreen_cache()
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    client = TestClient(create_app())

    protected_response = client.get("/v1/web/bigscreen-snapshot")
    web_response = client.get("/web-v2/snapshot")

    assert protected_response.status_code == 200
    assert web_response.status_code == 200
    assert protected_response.json()["mode"] == "read-only"
    assert web_response.json()["mode"] == "read-only"


def test_bigscreen_static_entry_is_served_if_built():
    static_index = Path("src/quant_lab/web/bigscreen_static/index.html")
    if not static_index.is_file():
        return

    response = TestClient(create_app()).get("/web-v2")

    assert response.status_code == 200
    assert "quant-lab CONTROL CENTER" in response.text


def test_web_v2_legacy_redirects_to_streamlit_port(monkeypatch):
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    monkeypatch.setenv("QUANT_LAB_LEGACY_WEB_PORT", "8501")
    client = TestClient(create_app(), base_url="http://qyun2.hrhome.top:8027")

    response = client.get("/web-v2/legacy", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "http://qyun2.hrhome.top:8501/"


def test_web_v2_expert_pack_generate_submits_read_only_job(monkeypatch, tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)

    calls: list[dict[str, str]] = []

    def fake_start_export_job(*, export_date, lake_root, exports_root):
        calls.append(
            {
                "export_date": export_date,
                "lake_root": str(lake_root),
                "exports_root": str(exports_root),
            }
        )
        return {
            "state": "running",
            "export_date": export_date,
            "trigger": "request_file",
            "request_path": str(exports_root / ".quant_lab_web_export_request.json"),
        }

    monkeypatch.setattr(
        "quant_lab.web.pages.expert_exports._start_export_job",
        fake_start_export_job,
    )

    response = TestClient(create_app()).post("/web-v2/expert-pack/generate")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "read_only_export"
    assert payload["live_order_effect"] == "none"
    assert payload["state"] == "running"
    assert calls
    assert calls[0]["lake_root"] == str(lake)
    assert calls[0]["exports_root"] == str(tmp_path / "exports")


def test_web_v2_expert_pack_status_and_download(monkeypatch, tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    exports = tmp_path / "exports"
    pack = exports / "quant_lab_expert_pack_2026-06-05_120000.zip"
    exports.mkdir(parents=True)
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"status": "OK"}))
        archive.writestr("data_quality.json", json.dumps({"status": "OK"}))
        archive.writestr("expert_questions.md", "下一步看什么？\n")

    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    client = TestClient(create_app())

    status_response = client.get("/web-v2/expert-pack/status?export_date=2026-06-05")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["latest_pack_name"] == pack.name
    assert status_payload["latest_download_url"] == f"/web-v2/expert-pack/download/{pack.name}"
    assert status_payload["packs"][0]["download_url"] == f"/web-v2/expert-pack/download/{pack.name}"

    download_response = client.get(f"/web-v2/expert-pack/download/{pack.name}")
    assert download_response.status_code == 200
    assert download_response.headers["content-type"].startswith("application/zip")

    traversal_response = client.get("/web-v2/expert-pack/download/..%5Csecret.zip")
    assert traversal_response.status_code == 404


def test_web_v2_expert_pack_status_keeps_latest_available_when_requested_date_missing(
    monkeypatch,
    tmp_path,
):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    exports = tmp_path / "exports"
    pack = exports / "quant_lab_expert_pack_2026-06-05_120000.zip"
    exports.mkdir(parents=True)
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"status": "OK"}))

    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)

    status_response = TestClient(create_app()).get(
        "/web-v2/expert-pack/status?export_date=2026-06-06"
    )
    payload = status_response.json()

    assert status_response.status_code == 200
    assert payload["state"] == "missing_requested_date"
    assert payload["requested_date_pack"] is None
    assert payload["latest_pack_name"] == pack.name
    assert payload["latest_pack_is_requested_date"] is False
    assert payload["latest_download_url"] == f"/web-v2/expert-pack/download/{pack.name}"


def test_bigscreen_exports_payload_separates_job_state_from_data_quality():
    pack_path = "/var/lib/quant-lab/exports/quant_lab_expert_pack_2026-06-05_120000.zip"

    payload = _exports_payload(
        {
            "latest_pack": pack_path,
            "packs": pl.DataFrame(
                [
                    {
                        "path": pack_path,
                        "name": "quant_lab_expert_pack_2026-06-05_120000.zip",
                        "size_bytes": 123,
                    }
                ]
            ),
            "manifest_summary": {},
            "data_quality_summary": {"status": "CRITICAL", "warning_count": 7},
            "expert_questions": [],
        }
    )

    assert payload["job_state"] == "pack_available"
    assert payload["data_quality_status"] == "CRITICAL"
    assert payload["data_quality_warning_count"] == 7
