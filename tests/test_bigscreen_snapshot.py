from __future__ import annotations

import json
import os
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
from fastapi.testclient import TestClient

import quant_lab.api.main as api_main
import quant_lab.web.bigscreen as bigscreen_module
from quant_lab.api.main import create_app
from quant_lab.data.lake import write_market_bars, write_parquet_dataset
from quant_lab.web.bigscreen import (
    _exports_payload,
    bigscreen_snapshot,
    bigscreen_snapshot_with_meta,
    clear_bigscreen_cache,
)


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


def test_bigscreen_snapshot_kpis_include_market_bar_close_time(tmp_path):
    lake = tmp_path / "lake"
    opened_at = datetime.now(UTC) - timedelta(hours=1, minutes=5)
    write_market_bars(
        lake,
        [
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": opened_at,
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
    clear_bigscreen_cache()

    payload = bigscreen_snapshot(lake)

    assert payload["kpis"]["latest_market_bar_ts"]
    assert payload["kpis"]["latest_market_bar_close_ts"]
    assert payload["data_health"]["latest_market_bar_close_ts"]
    assert payload["kpis"]["market_delay_seconds"] < 10 * 60


def test_bigscreen_snapshot_cache_covers_frontend_refresh_interval(monkeypatch, tmp_path):
    clear_bigscreen_cache()
    monkeypatch.delenv("QUANT_LAB_BIGSCREEN_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.delenv("QUANT_LAB_BIGSCREEN_STALE_GRACE_SECONDS", raising=False)
    monkeypatch.setenv("QUANT_LAB_BIGSCREEN_ASYNC_REFRESH", "0")
    now = 1000.0
    monkeypatch.setattr(bigscreen_module.time, "monotonic", lambda: now)

    first = bigscreen_snapshot(tmp_path / "lake")
    now = 1034.0
    second = bigscreen_snapshot(tmp_path / "lake")
    now = 1036.0
    third = bigscreen_snapshot(tmp_path / "lake")
    now = 1217.0
    fourth = bigscreen_snapshot(tmp_path / "lake")
    now = 2837.0
    fifth = bigscreen_snapshot(tmp_path / "lake")

    assert second is first
    assert third is not first
    assert fourth is not third
    assert fifth is not fourth


def test_bigscreen_strategy_flow_counts_use_full_advisory_not_display_sample(tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    created_at = datetime(2026, 6, 6, 13, 0, tzinfo=UTC)
    rows = []
    for index in range(532):
        if index < 5:
            decision = "PAPER_READY"
            mode = "paper"
        elif index < 9:
            decision = "KEEP_SHADOW"
            mode = "shadow"
        elif index < 12:
            decision = "RESEARCH_ONLY"
            mode = "research"
        else:
            decision = "KILL"
            mode = "none"
        row_ts = created_at + timedelta(seconds=index)
        rows.append(
            {
                "decision": decision,
                "recommended_mode": mode,
                "strategy_candidate": f"candidate_{index:03d}",
                "symbol": "BTC-USDT",
                "horizon_hours": 24,
                "avg_net_bps": float(index),
                "p25_net_bps": float(index) - 5.0,
                "complete_sample_count": 50 + index,
                "created_at": row_ts,
                "as_of_ts": row_ts,
            }
        )
    write_parquet_dataset(
        pl.DataFrame(rows),
        lake / "gold" / "strategy_opportunity_advisory",
    )

    payload = bigscreen_snapshot(lake)

    assert payload["strategy_flow"]["counts"] == {
        "research": 3,
        "shadow": 4,
        "paper": 5,
        "kill": 520,
    }
    assert payload["strategy_flow"]["top_live_candidates"]
    assert payload["strategy_flow"]["top_live_candidates"][0]["recommended_mode"] == "paper"
    assert payload["strategy_flow"]["top_candidates"][0]["decision"] == "PAPER_READY"


def test_bigscreen_data_matrix_shows_proxy_only_live_cost_diagnostics(tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    created_at = datetime.now(UTC)
    day = created_at.date().isoformat()
    common = {
        "regime": "all",
        "notional_bucket": "all",
        "fee_bps_p75": 4.0,
        "spread_bps_p75": 2.0,
        "slippage_bps_p75": 3.0,
        "total_cost_bps_p75": 9.0,
        "created_at": created_at,
        "day": day,
    }
    rows = [
        {
            **common,
            "symbol": "BTC-USDT",
            "source": "actual_okx_fills_and_bills",
            "cost_source": "actual_okx_fills_and_bills",
            "sample_count": 40,
            "actual_fill_count": 40,
            "mixed_fill_count": 0,
            "proxy_sample_count": 0,
            "fallback_level": "NONE",
        },
        *[
            {
                **common,
                "symbol": symbol,
                "source": "public_spread_proxy",
                "cost_source": "public_spread_proxy",
                "sample_count": 100,
                "actual_fill_count": 0,
                "mixed_fill_count": 0,
                "proxy_sample_count": 100,
                "fallback_level": "PUBLIC_SPREAD_PROXY",
            }
            for symbol in ["BNB-USDT", "ETH-USDT", "SOL-USDT"]
        ],
    ]
    write_parquet_dataset(pl.DataFrame(rows), lake / "gold" / "cost_bucket_daily")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": day,
                    "status": "warning",
                    "actual_rows": 0,
                    "mixed_rows": 0,
                    "proxy_rows": 33,
                    "global_default_rows": 0,
                    "hard_fallback_ratio": 0.0,
                    "soft_fallback_ratio": 1.0,
                    "proxy_only_count": 33,
                    "created_at": created_at,
                    "warnings_json": '["cost_soft_fallback"]',
                }
            ]
        ),
        lake / "gold" / "cost_health_daily",
    )

    payload = bigscreen_snapshot(lake)
    matrix = {row["symbol"]: row for row in payload["data_matrix"]["rows"]}

    assert matrix["BTC-USDT"]["cost"]["status"] == "OK"
    assert matrix["BTC-USDT"]["cost"]["source"] == "actual_okx_fills_and_bills"
    for symbol in ["BNB-USDT", "ETH-USDT", "SOL-USDT"]:
        assert matrix[symbol]["cost"]["status"] == "WARNING"
        assert matrix[symbol]["cost"]["source"] == "public_spread_proxy"
        assert matrix[symbol]["cost"]["coverage_status"] == "WARNING"
    coverage = {row["symbol"]: row for row in payload["cost"]["live_universe_cost_coverage"]}
    assert coverage["SOL-USDT"]["effective_cost_source"] == "public_spread_proxy"
    assert coverage["SOL-USDT"]["actual_or_mixed_covered"] is False
    assert payload["cost"]["soft_fallback_ratio"] == 1.0
    assert not any(action["title"] == "成本软回退偏高" for action in payload["actions"])


def test_bigscreen_cost_payload_exposes_bootstrap_probe_rows(tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    created_at = datetime.now(UTC).isoformat()
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "SOL-USDT",
                    "regime": "realized",
                    "notional_bucket": "all",
                    "source": "bootstrap_cost_probe",
                    "cost_source": "bootstrap_cost_probe",
                    "sample_count": 2,
                    "total_cost_bps_p75": 12.0,
                    "created_at": created_at,
                    "day": "2026-06-23",
                }
            ]
        ),
        lake / "gold" / "cost_bucket_daily",
    )

    payload = bigscreen_snapshot(lake)

    assert payload["cost"]["bootstrap_probe_rows"] == 1
    assert payload["cost"]["actual_rows"] == 0


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


def test_bigscreen_snapshot_exposes_cost_probe_p3_preflight(tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "date": "2026-06-18",
                    "status": "OK",
                    "latest_bundle_ts": "2026-06-18T14:54:31Z",
                    "latest_bundle_sha256": "abc123",
                }
            ]
        ),
        lake / "gold" / "strategy_health_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_ts": "2026-06-18T14:54:31Z",
                    "ingest_ts": "2026-06-18T14:55:00Z",
                    "generated_at_utc": "2026-06-18T14:54:31Z",
                    "source_path_inside_bundle": "summaries/cost_probe_p3_preflight.json",
                    "state": "NOT_READY",
                    "ready_to_request_manual_live_probe": False,
                    "approved_live_order_execution": False,
                    "live_order_effect": "none_preflight_only_no_order",
                    "raw_payload_json": "{}",
                }
            ]
        ),
        lake / "silver" / "v5_cost_probe_p3_preflight",
    )

    payload = bigscreen_snapshot(lake)

    p3 = payload["v5"]["cost_probe_p3_preflight"]
    assert p3["state"] == "NOT_READY"
    assert p3["approved_live_order_execution"] is False
    assert p3["live_order_effect"] == "none_preflight_only_no_order"


def test_bigscreen_snapshot_uses_latest_v5_bundle_with_same_day_rows(tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    reports = lake / "reports"
    reports.mkdir(parents=True)
    (reports / "v5_enforce_readiness.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-06-25T04:00:00Z",
                "readiness_status": "ADVISORY_READY",
                "veto_status": "VETO_READY",
                "entry_status": "ENTRY_BLOCKED",
                "scale_status": "SCALE_BLOCKED",
                "blocked_reasons": [],
                "warning_reasons": [
                    "actual_or_mixed_cost_coverage_live_universe",
                ],
                "entry_blocked_reasons": [
                    "actual_or_mixed_cost_coverage_live_universe",
                ],
                "scale_blocked_reasons": [
                    "actual_or_mixed_cost_coverage_live_universe",
                ],
            }
        ),
        encoding="utf-8",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "bundle_ts": "2026-06-25T03:27:25Z",
                    "ingest_ts": "2026-06-25T03:28:00Z",
                    "bundle_name": "old.tar.gz",
                }
            ]
        ),
        lake / "bronze" / "strategy_telemetry" / "v5" / "bundle_manifest",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "date": "2026-06-25",
                    "status": "OK",
                    "latest_bundle_ts": "2026-06-25T03:27:25Z",
                    "latest_bundle_sha256": "old-sha",
                },
                {
                    "date": "2026-06-25",
                    "status": "OK",
                    "latest_bundle_ts": "2026-06-25T03:56:10Z",
                    "latest_bundle_sha256": "new-sha",
                },
            ]
        ),
        lake / "gold" / "strategy_health_daily",
    )

    payload = bigscreen_snapshot(lake)

    assert payload["kpis"]["latest_v5_bundle_ts"] == "2026-06-25T03:56:10Z"
    assert payload["v5"]["latest_bundle_sha256"] == "new-sha"
    readiness = payload["v5"]["current_enforce_readiness"]
    assert readiness["veto_status"] == "VETO_READY"
    assert readiness["entry_status"] == "ENTRY_BLOCKED"
    assert readiness["scale_status"] == "SCALE_BLOCKED"


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
        archive.writestr(
            "reports/fast_microstructure_forward_test.csv",
            "\n".join(
                [
                    (
                        "generated_at,feature_name,symbol,regime,horizon_hours,"
                        "sample_count,effective_sample_count,rank_ic,long_short_bps,"
                        "p25_net_bps,hit_rate,recent_7d_score,walk_forward_oos_score,"
                        "oos_validation_pass,block_bootstrap_ci_low_bps,"
                        "block_bootstrap_ci_high_bps,purge_embargo_hours,lookback_bars,"
                        "build_elapsed_ms,recommendation,"
                        "data_leakage_check,live_order_effect"
                    ),
                    (
                        "2026-06-06T13:00:00Z,orderbook_imbalance_1m,SOL-USDT,"
                        "TREND_UP,8,64,56,0.21,58.5,2.1,0.64,42.0,18.0,"
                        "true,4.0,72.0,8,2000,12.4,"
                        "FORWARD_VALIDATION_PASS,"
                        "pass_future_prices_used_only_for_labels;purge_embargo_by_horizon,"
                        "read_only_no_live_order"
                    ),
                    (
                        "2026-06-06T13:00:00Z,cvd_5m,BTC-USDT,SIDEWAYS,4,12,"
                        "8,0.02,4.5,-8.0,0.51,3.0,,false,,,4,2000,12.4,"
                        "NEEDS_MORE_FORWARD_SAMPLES,"
                        "pass_future_prices_used_only_for_labels;purge_embargo_by_horizon,"
                        "read_only_no_live_order"
                    ),
                ]
            ),
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
    assert factor_factory["paper_review_queue_count"] == 1
    assert factor_factory["strategy_bridge_candidate_count"] == 1
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
    fast_forward = payload["strategy_flow"]["fast_microstructure_forward"]
    assert fast_forward["live_order_effect"] == "read_only_no_live_order"
    assert fast_forward["row_count"] == 2
    assert fast_forward["pass_count"] == 1
    assert fast_forward["needs_more_samples_count"] == 1
    assert fast_forward["lookback_bars"] == "2000"
    assert fast_forward["top_passes"][0]["feature_name"] == "orderbook_imbalance_1m"
    assert fast_forward["top_passes"][0]["symbol"] == "SOL-USDT"


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
    monkeypatch.setattr(api_main, "json", object())
    client = TestClient(create_app())

    protected_response = client.get("/v1/web/bigscreen-snapshot")
    web_response = client.get("/web-v2/snapshot")
    protected_cached_response = client.get("/v1/web/bigscreen-snapshot")

    assert protected_response.status_code == 200
    assert web_response.status_code == 200
    assert protected_cached_response.status_code == 200
    assert protected_response.headers["content-type"] == "application/json; charset=utf-8"
    assert web_response.headers["content-type"] == "application/json; charset=utf-8"
    assert protected_response.headers["cache-control"] == "no-store, max-age=0"
    assert web_response.headers["cache-control"] == "no-store, max-age=0"
    assert protected_response.headers["pragma"] == "no-cache"
    assert web_response.headers["pragma"] == "no-cache"
    assert protected_response.headers["x-quant-lab-bigscreen-mode"] == "read-only"
    assert web_response.headers["x-quant-lab-bigscreen-mode"] == "read-only"
    assert protected_response.headers["x-quant-lab-bigscreen-cache-hit"] == "false"
    assert web_response.headers["x-quant-lab-bigscreen-cache-hit"] == "true"
    assert protected_cached_response.headers["x-quant-lab-bigscreen-cache-hit"] == "true"
    assert protected_cached_response.headers["x-quant-lab-api-cache-hit"] == "true"
    assert protected_cached_response.headers["x-quant-lab-response-cache-hit"] == "true"
    assert float(web_response.headers["x-quant-lab-bigscreen-cache-age-seconds"]) >= 0.0
    assert float(web_response.headers["x-quant-lab-bigscreen-cache-ttl-seconds"]) >= 0.0
    assert int(protected_response.headers["x-quant-lab-response-bytes"]) == len(
        protected_response.content
    )
    assert int(web_response.headers["x-quant-lab-response-bytes"]) == len(web_response.content)
    assert protected_response.json()["mode"] == "read-only"
    assert web_response.json()["mode"] == "read-only"


def test_bigscreen_snapshot_serves_stale_cache_while_refresh_is_deferred(
    monkeypatch,
    tmp_path,
):
    clear_bigscreen_cache()
    monkeypatch.setenv("QUANT_LAB_BIGSCREEN_CACHE_TTL_SECONDS", "0")
    monkeypatch.setenv("QUANT_LAB_BIGSCREEN_STALE_GRACE_SECONDS", "60")
    monkeypatch.setenv("QUANT_LAB_BIGSCREEN_ASYNC_REFRESH", "0")
    calls: list[Path] = []

    def fake_build(root: Path) -> dict[str, object]:
        calls.append(root)
        return {
            "mode": "read-only",
            "status": "OK",
            "build_count": len(calls),
        }

    monkeypatch.setattr(bigscreen_module, "_build_bigscreen_snapshot_payload", fake_build)

    lake = tmp_path / "lake"
    first_payload, first_meta = bigscreen_snapshot_with_meta(lake)
    cache_key = str(lake.resolve())
    with bigscreen_module._SNAPSHOT_CACHE_LOCK:
        created_at, payload = bigscreen_module._SNAPSHOT_CACHE[cache_key]
        bigscreen_module._SNAPSHOT_CACHE[cache_key] = (created_at - 1.0, payload)
    stale_payload, stale_meta = bigscreen_snapshot_with_meta(lake)

    assert first_payload["build_count"] == 1
    assert first_meta["cache_hit"] is False
    assert stale_payload["build_count"] == 1
    assert stale_meta["cache_hit"] is True
    assert stale_meta["cache_stale"] is True
    assert stale_meta["cache_refresh_scheduled"] is False
    assert len(calls) == 1


def test_bigscreen_snapshot_cache_invalidates_when_export_index_changes(
    monkeypatch,
    tmp_path,
):
    clear_bigscreen_cache()
    monkeypatch.setenv("QUANT_LAB_BIGSCREEN_CACHE_TTL_SECONDS", "300")
    calls: list[Path] = []

    def fake_build(root: Path) -> dict[str, object]:
        calls.append(root)
        return {
            "mode": "read-only",
            "status": "OK",
            "build_count": len(calls),
        }

    monkeypatch.setattr(bigscreen_module, "_build_bigscreen_snapshot_payload", fake_build)
    lake = tmp_path / "lake"
    exports = tmp_path / "exports"
    exports.mkdir()

    first_payload, first_meta = bigscreen_snapshot_with_meta(lake)
    (exports / "export_index.json").write_text(
        json.dumps({"packs": [], "manifest_summary": {"marker": "updated"}}),
        encoding="utf-8",
    )
    second_payload, second_meta = bigscreen_snapshot_with_meta(lake)

    assert first_payload["build_count"] == 1
    assert first_meta["cache_hit"] is False
    assert second_payload["build_count"] == 2
    assert second_meta["cache_hit"] is False
    assert len(calls) == 2


def test_bigscreen_snapshot_cache_invalidates_when_manual_export_status_changes(
    monkeypatch,
    tmp_path,
):
    clear_bigscreen_cache()
    monkeypatch.setenv("QUANT_LAB_BIGSCREEN_CACHE_TTL_SECONDS", "300")
    monkeypatch.setattr(bigscreen_module, "beijing_today", lambda now=None: date(2026, 6, 28))
    calls: list[Path] = []

    def fake_build(root: Path) -> dict[str, object]:
        calls.append(root)
        return {
            "mode": "read-only",
            "status": "OK",
            "build_count": len(calls),
        }

    monkeypatch.setattr(bigscreen_module, "_build_bigscreen_snapshot_payload", fake_build)
    lake = tmp_path / "lake"
    exports = tmp_path / "exports"
    exports.mkdir()

    first_payload, first_meta = bigscreen_snapshot_with_meta(lake)
    (exports / ".quant_lab_web_export_2026-06-28.json").write_text(
        json.dumps(
            {
                "state": "running",
                "trigger": "manual_web_request",
                "started_at": "2026-06-27T21:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    second_payload, second_meta = bigscreen_snapshot_with_meta(lake)

    assert first_payload["build_count"] == 1
    assert first_meta["cache_hit"] is False
    assert second_payload["build_count"] == 2
    assert second_meta["cache_hit"] is False
    assert len(calls) == 2


def test_bigscreen_snapshot_cache_invalidates_when_market_bar_health_changes(
    monkeypatch,
    tmp_path,
):
    clear_bigscreen_cache()
    monkeypatch.setenv("QUANT_LAB_BIGSCREEN_CACHE_TTL_SECONDS", "300")
    calls: list[Path] = []

    def fake_build(root: Path) -> dict[str, object]:
        calls.append(root)
        return {
            "mode": "read-only",
            "status": "OK",
            "build_count": len(calls),
        }

    monkeypatch.setattr(bigscreen_module, "_build_bigscreen_snapshot_payload", fake_build)
    lake = tmp_path / "lake"
    first_ts = datetime(2026, 6, 27, 14, tzinfo=UTC)
    second_ts = first_ts + timedelta(hours=1)

    write_market_bars(
        lake,
        [
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": first_ts,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1.0,
                "quote_volume": 100.5,
                "source": "test",
                "ingest_ts": first_ts + timedelta(hours=1, minutes=1),
            }
        ],
    )

    first_payload, first_meta = bigscreen_snapshot_with_meta(lake)
    write_market_bars(
        lake,
        [
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": second_ts,
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 2.0,
                "quote_volume": 203.0,
                "source": "test",
                "ingest_ts": second_ts + timedelta(hours=1, minutes=1),
            }
        ],
    )
    second_payload, second_meta = bigscreen_snapshot_with_meta(lake)

    assert first_payload["build_count"] == 1
    assert first_meta["cache_hit"] is False
    assert second_payload["build_count"] == 2
    assert second_meta["cache_hit"] is False
    assert len(calls) == 2


def test_bigscreen_static_entry_is_served_if_built():
    static_index = Path("src/quant_lab/web/bigscreen_static/index.html")
    if not static_index.is_file():
        return

    response = TestClient(create_app()).get("/web-v2")

    assert response.status_code == 200
    assert "quant-lab CONTROL CENTER" in response.text


def test_favicon_request_does_not_404():
    response = TestClient(create_app()).get("/favicon.ico")

    assert response.status_code == 204


def test_web_v2_mounts_v5_telemetry_card_on_one_primary_page():
    app_source = Path("frontend-bigscreen/src/App.tsx").read_text(encoding="utf-8")

    assert app_source.count("<V5Telemetry ") == 1
    assert app_source.count("<PerfConsumers ") == 1

    strategy_page = app_source.split('case "strategy":', 1)[1].split('case "data":', 1)[0]
    data_page = app_source.split('case "data":', 1)[1].split('case "ops":', 1)[0]
    ops_page = app_source.split('case "ops":', 1)[1].split("    }\n  })();", 1)[0]

    assert "<V5Telemetry " not in strategy_page
    assert "<PerfConsumers " not in data_page
    assert "<V5Telemetry " in ops_page
    assert "<PerfConsumers " in ops_page
    assert 'href="/docs"' not in app_source
    assert 'href="/v1/web/bigscreen-snapshot"' not in app_source
    assert 'href="/web-v2/snapshot"' in app_source
    assert "snapshotPacks" not in app_source
    assert "exports.latest_download_url" not in app_source
    assert "exports.job_state" not in app_source
    assert "历史包" in app_source
    assert "包数 {status?.pack_count" not in app_source


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
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["pragma"] == "no-cache"
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
    assert status_response.headers["cache-control"] == "no-store, max-age=0"
    assert status_response.headers["pragma"] == "no-cache"
    status_payload = status_response.json()
    assert status_payload["state"] == "manual_missing"
    assert status_payload["available_pack_name"] == pack.name
    assert status_payload["latest_pack_name"] is None
    assert status_payload["latest_download_url"] is None
    assert status_payload["packs"][0]["download_url"] == f"/web-v2/expert-pack/download/{pack.name}"

    download_response = client.get(f"/web-v2/expert-pack/download/{pack.name}")
    assert download_response.status_code == 200
    assert download_response.headers["content-type"].startswith("application/zip")

    traversal_response = client.get("/web-v2/expert-pack/download/..%5Csecret.zip")
    assert traversal_response.status_code == 404


def test_web_v2_expert_pack_status_filters_deleted_index_pack(monkeypatch, tmp_path):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    exports = tmp_path / "exports"
    pack = exports / "quant_lab_expert_pack_2026-06-05_120000.zip"
    deleted_pack = exports / "quant_lab_expert_pack_2026-06-05_110000.zip"
    exports.mkdir(parents=True)
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"status": "OK"}))
        archive.writestr("data_quality.json", json.dumps({"status": "OK"}))
        archive.writestr("expert_questions.md", "下一步看什么？\n")
    (exports / "export_index.json").write_text(
        json.dumps(
            {
                "latest_pack": str(pack),
                "packs": [
                    {
                        "path": str(pack),
                        "name": pack.name,
                        "size_bytes": pack.stat().st_size,
                        "modified_at": "2026-06-05T12:00:00Z",
                    },
                    {
                        "path": str(deleted_pack),
                        "name": deleted_pack.name,
                        "size_bytes": 1,
                        "modified_at": "2026-06-05T11:00:00Z",
                    },
                ],
                "manifest_summary": {"status": "OK"},
                "data_quality_summary": {"status": "OK"},
                "expert_questions": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    payload = TestClient(create_app()).get(
        "/web-v2/expert-pack/status?export_date=2026-06-05"
    ).json()

    assert payload["available_pack_name"] == pack.name
    assert payload["latest_pack_name"] is None
    assert payload["pack_count"] == 1
    assert [row["name"] for row in payload["packs"]] == [pack.name]


def test_web_v2_expert_pack_status_prefers_latest_requested_pack_over_stale_manual_status(
    monkeypatch,
    tmp_path,
):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    exports = tmp_path / "exports"
    exports.mkdir(parents=True)
    old_pack = exports / "quant_lab_expert_pack_2026-06-19_20260619T095826451309+0800.zip"
    new_pack = exports / "quant_lab_expert_pack_2026-06-19_20260619T155027688187+0800.zip"
    for pack in (old_pack, new_pack):
        with zipfile.ZipFile(pack, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"status": "OK"}))
            archive.writestr("data_quality.json", json.dumps({"status": "OK"}))
            archive.writestr("expert_questions.md", "下一步看什么？\n")
    old_time = datetime(2026, 6, 19, 2, tzinfo=UTC).timestamp()
    new_time = datetime(2026, 6, 19, 7, tzinfo=UTC).timestamp()
    os.utime(old_pack, (old_time, old_time))
    os.utime(new_pack, (new_time, new_time))
    (exports / ".quant_lab_web_export_2026-06-19.json").write_text(
        json.dumps(
            {
                "state": "succeeded",
                "zip_path": str(old_pack),
                "finished_at": "2026-06-19T02:03:31+00:00",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    response = TestClient(create_app()).get(
        "/web-v2/expert-pack/status?export_date=2026-06-19"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"]["zip_path"] == str(old_pack)
    assert payload["requested_date_pack_name"] == new_pack.name
    assert payload["available_pack_name"] == new_pack.name
    assert payload["manual_latest_pack_name"] == old_pack.name
    assert payload["latest_pack_name"] == new_pack.name
    assert payload["latest_pack_is_requested_date"] is True
    assert payload["latest_pack_source"] == "requested_date_pack"
    assert payload["latest_download_url"] == f"/web-v2/expert-pack/download/{new_pack.name}"
    assert (
        payload["manual_latest_download_url"]
        == f"/web-v2/expert-pack/download/{old_pack.name}"
    )
    assert payload["packs"][0]["name"] == new_pack.name


def test_web_v2_expert_pack_status_hides_previous_pack_while_running(
    monkeypatch,
    tmp_path,
):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    exports = tmp_path / "exports"
    exports.mkdir(parents=True)
    previous_pack = exports / "quant_lab_expert_pack_2026-06-20_20260620T002151627801+0800.zip"
    with zipfile.ZipFile(previous_pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"status": "OK"}))
        archive.writestr("data_quality.json", json.dumps({"status": "OK"}))
        archive.writestr("expert_questions.md", "下一步看什么？\n")
    previous_time = datetime(2026, 6, 19, 16, tzinfo=UTC).timestamp()
    os.utime(previous_pack, (previous_time, previous_time))
    request_path = exports / ".quant_lab_web_export_request.json"
    request_path.write_text("{}", encoding="utf-8")
    (exports / ".quant_lab_web_export_2026-06-20.json").write_text(
        json.dumps(
            {
                "state": "running",
                "trigger": "request_file",
                "request_path": str(request_path),
                "systemd_unit": "quant-lab-web-export-request.service",
                "started_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    response = TestClient(create_app()).get(
        "/web-v2/expert-pack/status?export_date=2026-06-20"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "running"
    assert payload["requested_date_pack_name"] == previous_pack.name
    assert payload["previous_pack_name"] == previous_pack.name
    assert payload["latest_pack_name"] is None
    assert payload["latest_pack_is_requested_date"] is False
    assert payload["latest_download_url"] is None
    assert "latest_size_bytes" not in payload


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
    assert payload["available_pack_name"] == pack.name
    assert payload["latest_pack_name"] is None
    assert payload["latest_pack_is_requested_date"] is False
    assert payload["latest_download_url"] is None


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


def test_bigscreen_snapshot_treats_history_export_as_available_not_current(
    tmp_path,
    monkeypatch,
):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    lake.mkdir()
    exports = tmp_path / "exports"
    exports.mkdir()
    pack_path = exports / "quant_lab_expert_pack_2026-06-05_120000.zip"
    with zipfile.ZipFile(pack_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"export_date": "2026-06-05"}))
        archive.writestr(
            "data_quality.json",
            json.dumps({"status": "CRITICAL", "warning_count": 3}),
        )
        archive.writestr("expert_questions.md", "历史包问题\n")
    (exports / "export_index.json").write_text(
        json.dumps(
            {
                "latest_pack": str(pack_path),
                "packs": [
                    {
                        "path": str(pack_path),
                        "name": pack_path.name,
                        "size_bytes": 123,
                        "modified_at": "2026-06-05T12:00:00Z",
                    }
                ],
                "manifest_summary": {"export_date": "2026-06-05"},
                "data_quality_summary": {"status": "CRITICAL", "warning_count": 3},
                "expert_questions": ["历史包问题"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bigscreen_module, "beijing_today", lambda now=None: date(2026, 6, 6))

    history = bigscreen_module.readers.expert_export_summary(exports)
    summary = bigscreen_module._web_v2_export_summary_from_history(
        lake,
        history,
        datetime(2026, 6, 5, 13, tzinfo=UTC),
    )
    payload = _exports_payload(summary)
    actions = bigscreen_module._build_actions(
        overview={},
        data_health={},
        cost={},
        v5={"latest": {"kill_switch_enabled": False, "reconcile_ok": True}},
        web_events=[],
        exports=summary,
        legacy_anomalies={"items": []},
    )

    assert summary["latest_pack"] is None
    assert summary["available_pack"] == str(pack_path.resolve())
    assert summary["data_quality_summary"] == {}
    assert payload["latest_download_url"] is None
    assert payload["available_pack_name"] == pack_path.name
    assert payload["job_state"] == "manual_missing"
    assert not any(action["source"] == "expert_export_summary" for action in actions)


def test_bigscreen_snapshot_uses_latest_current_date_pack_over_stale_manual_status(
    tmp_path,
    monkeypatch,
):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    lake.mkdir()
    exports = tmp_path / "exports"
    exports.mkdir()
    old_pack = exports / "quant_lab_expert_pack_2026-06-19_20260619T095826451309+0800.zip"
    new_pack = exports / "quant_lab_expert_pack_2026-06-19_20260619T155027688187+0800.zip"
    for pack in (old_pack, new_pack):
        with zipfile.ZipFile(pack, "w") as archive:
            archive.writestr(
                "manifest.json",
                json.dumps({"export_date": "2026-06-19", "authoritative_snapshot": True}),
            )
            archive.writestr("data_quality.json", json.dumps({"status": "OK"}))
            archive.writestr("expert_questions.md", "最新问题\n")
    old_time = datetime(2026, 6, 19, 2, tzinfo=UTC).timestamp()
    new_time = datetime(2026, 6, 19, 7, tzinfo=UTC).timestamp()
    os.utime(old_pack, (old_time, old_time))
    os.utime(new_pack, (new_time, new_time))
    (exports / ".quant_lab_web_export_2026-06-19.json").write_text(
        json.dumps(
            {
                "state": "succeeded",
                "zip_path": str(old_pack),
                "finished_at": "2026-06-19T02:03:31+00:00",
            }
        ),
        encoding="utf-8",
    )
    (exports / "export_index.json").write_text(
        json.dumps(
            {
                "latest_pack": str(new_pack),
                "packs": [
                    {
                        "path": str(new_pack),
                        "name": new_pack.name,
                        "selected_v5_bundle_manifest_bundle_name": (
                            "v5_live_followup_bundle_20260619T070000Z.tar.gz"
                        ),
                        "size_bytes": 456,
                        "modified_at": "2026-06-19T07:00:00Z",
                    },
                    {
                        "path": str(old_pack),
                        "name": old_pack.name,
                        "size_bytes": 123,
                        "modified_at": "2026-06-19T02:00:00Z",
                    },
                ],
                "manifest_summary": {"export_date": "2026-06-19"},
                "data_quality_summary": {"status": "OK"},
                "expert_questions": ["最新问题"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bigscreen_module, "beijing_today", lambda now=None: date(2026, 6, 19))

    history = bigscreen_module.readers.expert_export_summary(exports)
    summary = bigscreen_module._web_v2_export_summary_from_history(
        lake,
        history,
        datetime(2026, 6, 19, 9, tzinfo=UTC),
    )
    payload = _exports_payload(summary)

    assert summary["manual_latest_pack_name"] == old_pack.name
    assert Path(summary["latest_pack"]).name == new_pack.name
    assert summary["latest_pack_source"] == "requested_date_pack"
    assert payload["latest_pack_name"] == new_pack.name
    assert (
        payload["latest_pack_v5_bundle_name"]
        == "v5_live_followup_bundle_20260619T070000Z.tar.gz"
    )
    assert payload["latest_pack_v5_bundle_ts"] == "2026-06-19T07:00:00Z"
    assert payload["latest_download_url"] == f"/web-v2/expert-pack/download/{new_pack.name}"


def test_bigscreen_action_warns_when_expert_pack_v5_bundle_lags_current_v5():
    pack_path = "/var/lib/quant-lab/exports/quant_lab_expert_pack_2026-06-28_120000.zip"
    actions = bigscreen_module._build_actions(
        overview={},
        data_health={},
        cost={},
        v5={"latest": {"latest_bundle_ts": "2026-06-28T13:22:51Z"}},
        web_events=[],
        exports={
            "latest_pack": pack_path,
            "latest_pack_source": "requested_date_pack",
            "packs": pl.DataFrame(
                [
                    {
                        "path": pack_path,
                        "name": Path(pack_path).name,
                        "selected_v5_bundle_manifest_bundle_name": (
                            "v5_live_followup_bundle_20260628T120533Z.tar.gz"
                        ),
                    }
                ]
            ),
            "data_quality_summary": {"status": "OK", "warning_count": 0},
        },
    )

    action = next(
        item for item in actions if item["source"] == "expert_export_summary.v5_bundle_lag"
    )
    assert action["severity"] == "WARNING"
    assert action["title"] == "专家包落后 V5 遥测"
    assert "2026-06-28T13:22:51Z" in action["summary"]
    assert action["drilldown"] == "/exports"


def test_bigscreen_action_does_not_warn_for_current_expert_pack_v5_bundle():
    pack_path = "/var/lib/quant-lab/exports/quant_lab_expert_pack_2026-06-28_120000.zip"
    actions = bigscreen_module._build_actions(
        overview={},
        data_health={},
        cost={},
        v5={"latest": {"latest_bundle_ts": "2026-06-28T13:22:51Z"}},
        web_events=[],
        exports={
            "latest_pack": pack_path,
            "packs": pl.DataFrame(
                [
                    {
                        "path": pack_path,
                        "name": Path(pack_path).name,
                        "selected_v5_bundle_manifest_bundle_name": (
                            "v5_live_followup_bundle_20260628T131155Z.tar.gz"
                        ),
                    }
                ]
            ),
            "data_quality_summary": {"status": "OK", "warning_count": 0},
        },
    )

    assert not any(
        item["source"] == "expert_export_summary.v5_bundle_lag" for item in actions
    )


def test_bigscreen_snapshot_promotes_export_data_quality_warning(tmp_path, monkeypatch):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    lake.mkdir()
    exports = tmp_path / "exports"
    exports.mkdir()
    pack_path = exports / "quant_lab_expert_pack_2026-06-05_120000.zip"
    data_quality = {
        "status": "WARN",
        "warning_count": 2,
        "warnings": ["cost_soft_fallback_ratio: high", "okx_ws_universe_incomplete"],
        "failures": [],
    }
    with zipfile.ZipFile(pack_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"export_date": "2026-06-05"}))
        archive.writestr("data_quality.json", json.dumps(data_quality))
        archive.writestr("expert_questions.md", "")
    (exports / "export_index.json").write_text(
        json.dumps(
            {
                "latest_pack": str(pack_path),
                "packs": [
                    {
                        "path": str(pack_path),
                        "name": pack_path.name,
                        "size_bytes": 123,
                        "modified_at": "2026-06-05T12:00:00Z",
                    }
                ],
                "manifest_summary": {"export_date": "2026-06-05"},
                "data_quality_summary": data_quality,
                "expert_questions": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (exports / ".quant_lab_web_export_2026-06-05.json").write_text(
        json.dumps(
            {
                "state": "succeeded",
                "zip_path": str(pack_path),
                "finished_at": "2026-06-05T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bigscreen_module, "beijing_today", lambda now=None: date(2026, 6, 5))
    now = datetime.now(UTC)
    monkeypatch.setattr(
        bigscreen_module.readers,
        "data_health_summary",
        lambda _root: {
            "schema_violation_count": 0,
            "unclosed_bar_count": 0,
            "latest_market_bar_ts": now,
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        bigscreen_module.readers,
        "cost_model_summary",
        lambda _root: {"hard_fallback_ratio": 0.0, "soft_fallback_ratio": 0.0, "warnings": []},
    )
    monkeypatch.setattr(
        bigscreen_module,
        "_safe_strategy_summary",
        lambda _root: {
            "strategy_opportunity_advisory": pl.DataFrame(),
            "alpha_discovery_board": pl.DataFrame(),
            "factor_strategy_bridge_candidates": pl.DataFrame(),
            "fast_microstructure_forward_test": pl.DataFrame(),
            "strategy_counts": {},
            "discovery_counts": {},
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        bigscreen_module.readers,
        "market_regime_summary",
        lambda _root: {"warnings": []},
    )
    monkeypatch.setattr(
        bigscreen_module.readers,
        "okx_collector_summary",
        lambda _root: {"warnings": []},
    )
    monkeypatch.setattr(
        bigscreen_module.readers,
        "v5_telemetry_summary",
        lambda _root: {
            "latest": {
                "latest_bundle_ts": now,
                "kill_switch_enabled": False,
                "reconcile_ok": True,
                "ledger_ok": True,
            },
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        bigscreen_module.readers,
        "strategy_consumer_summary",
        lambda _root: {"permissions": {}, "warnings": []},
    )
    monkeypatch.setattr(bigscreen_module.perf, "recent_events", lambda limit=50: [])
    monkeypatch.setattr(bigscreen_module, "_safe_api_metrics", lambda _root: {})

    payload = bigscreen_snapshot(lake)

    assert payload["exports"]["data_quality_status"] == "WARN"
    assert payload["status"] == "WARNING"
    assert payload["health_score"] < 100
    assert any("expert_pack_data_quality_warning:" in item for item in payload["warnings"])
    assert any(
        action["source"] == "expert_export_summary" and action["severity"] == "WARNING"
        for action in payload["actions"]
    )


def test_bigscreen_actions_skip_read_only_cost_coverage_advisory():
    actions = bigscreen_module._build_actions(
        overview={},
        data_health={},
        cost={
            "hard_fallback_ratio": 0.0,
            "soft_fallback_ratio": 1.0,
            "live_universe_cost_coverage": [
                {
                    "symbol": "BTC-USDT",
                    "coverage_status": "WARNING",
                    "coverage_reason": "stale_actual_or_mixed_no_fresh_live_anchor",
                    "live_order_effect": "read_only_no_live_order",
                },
                {
                    "symbol": "ETH-USDT",
                    "coverage_status": "WARNING",
                    "coverage_reason": "public_proxy_only_no_live_actual_anchor",
                    "live_order_effect": "read_only_no_live_order",
                },
            ],
        },
        v5={"latest": {"kill_switch_enabled": False, "reconcile_ok": True}},
        web_events=[],
        exports={
            "latest_pack": "/tmp/quant_lab_expert_pack_2026-06-18_120000.zip",
            "data_quality_summary": {"status": "PASS", "warnings": [], "failures": []},
        },
        legacy_anomalies={"items": []},
    )

    assert not any(action["source"] == "cost_model_summary" for action in actions)


def test_bigscreen_actions_skip_read_only_export_cost_advisory():
    actions = bigscreen_module._build_actions(
        overview={},
        data_health={},
        cost={
            "hard_fallback_ratio": 0.0,
            "soft_fallback_ratio": 1.0,
            "live_universe_cost_coverage": [
                {
                    "symbol": "BTC-USDT",
                    "coverage_status": "WARNING",
                    "live_order_effect": "read_only_no_live_order",
                }
            ],
        },
        v5={"latest": {"kill_switch_enabled": False, "reconcile_ok": True}},
        web_events=[],
        exports={
            "latest_pack": "/tmp/quant_lab_expert_pack_2026-06-18_120000.zip",
            "data_quality_summary": {
                "status": "WARN",
                "warning_count": 3,
                "warnings": [
                    "cost_soft_fallback_ratio: soft_fallback_count=32; ratio=100.00%; "
                    "proxy_only_count=32",
                    "live_universe_stale_actual_or_mixed_cost: "
                    "stale_actual_or_mixed_symbols=['BTC-USDT']; "
                    "coverage_status=WARNING; coverage_rate=0.00%; "
                    "live_order_effect=read_only_no_live_order",
                    "quant_lab_enforce_readiness: readiness_status=BLOCKED; "
                    "blocked=['actual_or_mixed_cost_coverage_live_universe']; "
                    "warnings=['actual_or_mixed_cost_coverage_research_universe']; "
                    "live_order_effect=read_only_no_live_order",
                ],
                "failures": [],
            },
        },
        legacy_anomalies={"items": []},
    )

    assert not any(action["source"] == "expert_export_summary" for action in actions)


def test_bigscreen_market_bar_delay_warning_before_registry_stale_threshold(monkeypatch):
    now = datetime(2026, 6, 27, 14, 30, tzinfo=UTC)
    data_health = {
        "latest_market_bar_ts": now - timedelta(hours=3, minutes=5),
        "schema_violation_count": 0,
        "unclosed_bar_count": 0,
    }

    monkeypatch.delenv("QUANT_LAB_MARKET_BAR_WARNING_DELAY_SECONDS", raising=False)
    monkeypatch.delenv("QUANT_LAB_MARKET_BAR_CRITICAL_DELAY_SECONDS", raising=False)

    warnings = bigscreen_module._system_warnings([], {}, data_health, now)
    status = bigscreen_module._status_from_inputs(
        {"status": "OK"},
        data_health,
        {},
        {"latest": {}},
        warnings,
        {},
        generated_at=now,
    )

    assert status == "WARNING"
    assert bigscreen_module._market_bar_status(data_health, now) == "WARNING"
    assert any("market_bar_freshness_warning:" in warning for warning in warnings)
    assert any("latest_close=" in warning for warning in warnings)


def test_bigscreen_market_bar_delay_uses_bar_close_time(monkeypatch):
    now = datetime(2026, 6, 27, 14, 30, tzinfo=UTC)
    data_health = {
        "latest_market_bar_ts": now - timedelta(hours=2, minutes=5),
        "schema_violation_count": 0,
        "unclosed_bar_count": 0,
    }

    monkeypatch.delenv("QUANT_LAB_MARKET_BAR_WARNING_DELAY_SECONDS", raising=False)
    monkeypatch.delenv("QUANT_LAB_MARKET_BAR_CRITICAL_DELAY_SECONDS", raising=False)

    warnings = bigscreen_module._system_warnings([], {}, data_health, now)

    assert bigscreen_module._market_bar_status(data_health, now) == "OK"
    assert warnings == []
    assert bigscreen_module._market_bar_delay_seconds(data_health, now) < 2 * 60 * 60


def test_bigscreen_market_bar_delay_critical_after_registry_stale_threshold(monkeypatch):
    now = datetime(2026, 6, 27, 14, 30, tzinfo=UTC)
    data_health = {
        "latest_market_bar_ts": now - timedelta(hours=4, minutes=5),
        "schema_violation_count": 0,
        "unclosed_bar_count": 0,
    }

    monkeypatch.delenv("QUANT_LAB_MARKET_BAR_WARNING_DELAY_SECONDS", raising=False)
    monkeypatch.delenv("QUANT_LAB_MARKET_BAR_CRITICAL_DELAY_SECONDS", raising=False)

    warnings = bigscreen_module._system_warnings([], {}, data_health, now)
    status = bigscreen_module._status_from_inputs(
        {"status": "OK"},
        data_health,
        {},
        {"latest": {}},
        warnings,
        {},
        generated_at=now,
    )

    assert status == "CRITICAL"
    assert bigscreen_module._market_bar_status(data_health, now) == "CRITICAL"
    assert any("market_bar_freshness_critical:" in warning for warning in warnings)


def test_bigscreen_actions_skip_entry_or_scale_readiness_advisory():
    actions = bigscreen_module._build_actions(
        overview={},
        data_health={},
        cost={},
        v5={"latest": {"kill_switch_enabled": False, "reconcile_ok": True}},
        web_events=[],
        exports={
            "latest_pack": "/tmp/quant_lab_expert_pack_2026-06-22_120000.zip",
            "data_quality_summary": {
                "status": "WARN",
                "warning_count": 1,
                "warnings": [
                    "quant_lab_enforce_readiness: readiness_status=BLOCKED; "
                    "veto_status=VETO_READY; entry_status=ENTRY_BLOCKED; "
                    "scale_status=SCALE_BLOCKED; "
                    "blocked=['actual_or_mixed_cost_coverage_live_universe', 'fallback_rate']; "
                    "warnings=[]; live_order_effect=entry_or_scale_block_only"
                ],
                "failures": [],
            },
        },
        legacy_anomalies={"items": []},
    )

    assert not any(action["source"] == "expert_export_summary" for action in actions)


def test_bigscreen_system_warnings_skip_entry_or_scale_readiness_advisory():
    exports = {
        "data_quality_summary": {
            "status": "WARN",
            "warning_count": 1,
            "warnings": [
                "quant_lab_enforce_readiness: readiness_status=BLOCKED; "
                "veto_status=VETO_READY; entry_status=ENTRY_BLOCKED; scale_status=SCALE_BLOCKED; "
                "blocked=['actual_or_mixed_cost_coverage_live_universe', 'fallback_rate']; "
                "warnings=[]; live_order_effect=entry_or_scale_block_only"
            ],
            "failures": [],
        }
    }
    raw_warnings = bigscreen_module._export_quality_warnings(exports)
    raw_warnings.append("okx_collector_stale")

    system_warnings = bigscreen_module._system_warnings(raw_warnings, exports)

    assert system_warnings == ["okx_collector_stale"]


def test_bigscreen_snapshot_keeps_live_readiness_block_out_of_system_critical(
    tmp_path,
    monkeypatch,
):
    clear_bigscreen_cache()
    lake = tmp_path / "lake"
    lake.mkdir()
    exports = tmp_path / "exports"
    exports.mkdir()
    pack_path = exports / "quant_lab_expert_pack_2026-06-17_120000.zip"
    data_quality = {
        "status": "CRITICAL",
        "warning_count": 2,
        "warnings": [
            "cost_soft_fallback_ratio: soft_fallback_count=32",
            "quant_lab_enforce_readiness: readiness_status=BLOCKED; "
            "blocked=['actual_or_mixed_cost_coverage_live_universe']",
        ],
        "failures": [
            "quant_lab_enforce_readiness: readiness_status=BLOCKED; "
            "blocked=['actual_or_mixed_cost_coverage_live_universe']; "
            "warnings=['actual_or_mixed_cost_coverage_research_universe']"
        ],
    }
    with zipfile.ZipFile(pack_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"export_date": "2026-06-17"}))
        archive.writestr("data_quality.json", json.dumps(data_quality))
        archive.writestr("expert_questions.md", "")
    (exports / "export_index.json").write_text(
        json.dumps(
            {
                "latest_pack": str(pack_path),
                "packs": [
                    {
                        "path": str(pack_path),
                        "name": pack_path.name,
                        "size_bytes": 123,
                        "modified_at": "2026-06-17T12:00:00Z",
                    }
                ],
                "manifest_summary": {"export_date": "2026-06-17"},
                "data_quality_summary": data_quality,
                "expert_questions": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (exports / ".quant_lab_web_export_2026-06-17.json").write_text(
        json.dumps(
            {
                "state": "succeeded",
                "zip_path": str(pack_path),
                "finished_at": "2026-06-17T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bigscreen_module, "beijing_today", lambda now=None: date(2026, 6, 17))
    now = datetime.now(UTC)
    monkeypatch.setattr(
        bigscreen_module.readers,
        "data_health_summary",
        lambda _root: {
            "schema_violation_count": 0,
            "unclosed_bar_count": 0,
            "latest_market_bar_ts": now,
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        bigscreen_module.readers,
        "cost_model_summary",
        lambda _root: {"hard_fallback_ratio": 0.0, "soft_fallback_ratio": 0.0, "warnings": []},
    )
    monkeypatch.setattr(
        bigscreen_module,
        "_safe_strategy_summary",
        lambda _root: {
            "strategy_opportunity_advisory": pl.DataFrame(),
            "alpha_discovery_board": pl.DataFrame(),
            "factor_strategy_bridge_candidates": pl.DataFrame(),
            "fast_microstructure_forward_test": pl.DataFrame(),
            "strategy_counts": {},
            "discovery_counts": {},
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        bigscreen_module.readers,
        "market_regime_summary",
        lambda _root: {"warnings": []},
    )
    monkeypatch.setattr(
        bigscreen_module.readers,
        "okx_collector_summary",
        lambda _root: {"warnings": []},
    )
    monkeypatch.setattr(
        bigscreen_module.readers,
        "v5_telemetry_summary",
        lambda _root: {
            "latest": {
                "latest_bundle_ts": now,
                "kill_switch_enabled": False,
                "reconcile_ok": True,
                "ledger_ok": True,
            },
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        bigscreen_module.readers,
        "strategy_consumer_summary",
        lambda _root: {"permissions": {}, "warnings": []},
    )
    monkeypatch.setattr(bigscreen_module.perf, "recent_events", lambda limit=50: [])
    monkeypatch.setattr(bigscreen_module, "_safe_api_metrics", lambda _root: {})

    payload = bigscreen_snapshot(lake)

    assert payload["exports"]["data_quality_status"] == "CRITICAL"
    assert payload["status"] == "WARNING"
    assert payload["health_score"] >= 80
    assert any("expert_pack_data_quality_warning:" in item for item in payload["warnings"])
    assert not any("expert_pack_data_quality_critical:" in item for item in payload["warnings"])
    assert not any(
        action["source"] == "expert_export_summary"
        for action in payload["actions"]
    )
