from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    layer: str
    relative_path: Path
    owner: str
    description: str = ""
    producer: str = ""
    consumers: tuple[str, ...] = ()
    write_mode: str = "replace"
    required: bool = True
    min_rows: int = 1
    primary_key: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    schema: dict[str, str] = field(default_factory=dict)
    timestamp_column: str | None = None
    utc_timestamp_columns: tuple[str, ...] = ()
    closed_bar_column: str | None = None
    freshness_seconds: int | None = None
    retention_days: int | None = None
    quality_rules: tuple[str, ...] = ()

    def to_row(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "layer": self.layer,
            "path": str(self.relative_path),
            "owner": self.owner,
            "description": self.description,
            "producer": self.producer,
            "consumers_json": json.dumps(list(self.consumers), sort_keys=True),
            "write_mode": self.write_mode,
            "required": self.required,
            "min_rows": self.min_rows,
            "primary_key_json": json.dumps(list(self.primary_key), sort_keys=True),
            "required_columns_json": json.dumps(list(self.required_columns), sort_keys=True),
            "schema_json": json.dumps(self.schema, sort_keys=True),
            "timestamp_column": self.timestamp_column,
            "utc_timestamp_columns_json": json.dumps(
                list(self.utc_timestamp_columns),
                sort_keys=True,
            ),
            "closed_bar_column": self.closed_bar_column,
            "freshness_seconds": self.freshness_seconds,
            "retention_days": self.retention_days,
            "quality_rules_json": json.dumps(list(self.quality_rules), sort_keys=True),
        }


def core_dataset_specs() -> dict[str, DatasetSpec]:
    specs = [
        DatasetSpec(
            dataset_id="market_bar",
            layer="silver",
            relative_path=Path("silver") / "market_bar",
            owner="market-data",
            description="Closed OKX-first OHLCV bars used by features and research.",
            producer="okx-public-rest/ws",
            consumers=("features", "costs", "research", "web", "api"),
            write_mode="upsert",
            primary_key=("venue", "symbol", "timeframe", "ts"),
            required_columns=(
                "venue",
                "symbol",
                "market_type",
                "timeframe",
                "ts",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "source",
                "ingest_ts",
                "is_closed",
            ),
            schema={
                "ts": "datetime[UTC]",
                "ingest_ts": "datetime[UTC]",
                "is_closed": "boolean",
            },
            timestamp_column="ts",
            utc_timestamp_columns=("ts", "ingest_ts"),
            closed_bar_column="is_closed",
            freshness_seconds=3 * 60 * 60,
            retention_days=None,
            quality_rules=(
                "schema_required_columns",
                "primary_key_unique",
                "utc_timestamps",
                "closed_bar_only",
                "freshness",
            ),
        ),
        DatasetSpec(
            dataset_id="feature_value",
            layer="gold",
            relative_path=Path("gold") / "feature_value",
            owner="features",
            description="Published research features computed from closed market bars.",
            producer="qlab publish-features",
            consumers=("research", "web", "expert-export"),
            write_mode="upsert",
            primary_key=(
                "feature_set",
                "feature_name",
                "feature_version",
                "symbol",
                "timeframe",
                "ts",
            ),
            required_columns=(
                "feature_set",
                "feature_name",
                "feature_version",
                "symbol",
                "timeframe",
                "ts",
                "value",
                "created_at",
                "source",
                "is_valid",
            ),
            timestamp_column="created_at",
            utc_timestamp_columns=("ts", "created_at"),
            freshness_seconds=6 * 60 * 60,
            quality_rules=(
                "schema_required_columns",
                "primary_key_unique",
                "utc_timestamps",
                "freshness",
            ),
        ),
        DatasetSpec(
            dataset_id="feature_coverage_daily",
            layer="gold",
            relative_path=Path("gold") / "feature_coverage_daily",
            owner="features",
            producer="qlab publish-features",
            consumers=("web", "expert-export", "data-quality"),
            required_columns=(
                "day",
                "feature_name",
                "total_rows",
                "valid_rows",
                "coverage",
                "created_at",
            ),
            timestamp_column="created_at",
            utc_timestamp_columns=("created_at",),
            freshness_seconds=24 * 60 * 60,
            quality_rules=("schema_required_columns", "freshness"),
        ),
        DatasetSpec(
            dataset_id="feature_anomaly_daily",
            layer="gold",
            relative_path=Path("gold") / "feature_anomaly_daily",
            owner="features",
            producer="qlab publish-features",
            consumers=("web", "expert-export", "data-quality"),
            required=False,
            min_rows=0,
            required_columns=("day", "anomaly_type", "anomaly_count", "severity", "created_at"),
            timestamp_column="created_at",
            utc_timestamp_columns=("created_at",),
            freshness_seconds=24 * 60 * 60,
            quality_rules=("schema_required_columns", "freshness"),
        ),
        DatasetSpec(
            dataset_id="cost_bucket_daily",
            layer="gold",
            relative_path=Path("gold") / "cost_bucket_daily",
            owner="cost-model",
            producer="qlab calibrate-costs",
            consumers=("api", "v5", "web", "expert-export", "readiness"),
            write_mode="upsert",
            primary_key=("day", "symbol", "regime", "event_type", "notional_bucket"),
            required_columns=(
                "day",
                "symbol",
                "regime",
                "sample_count",
                "total_cost_bps_p50",
                "total_cost_bps_p75",
                "total_cost_bps_p90",
                "cost_source",
                "fallback_level",
                "created_at",
            ),
            timestamp_column="created_at",
            utc_timestamp_columns=("created_at",),
            freshness_seconds=24 * 60 * 60,
            quality_rules=(
                "schema_required_columns",
                "primary_key_unique",
                "freshness",
                "cost_fallback_visibility",
            ),
        ),
        DatasetSpec(
            dataset_id="cost_health_daily",
            layer="gold",
            relative_path=Path("gold") / "cost_health_daily",
            owner="cost-model",
            producer="qlab cost-health",
            consumers=("readiness", "web", "expert-export"),
            required_columns=(
                "day",
                "status",
                "actual_rows",
                "mixed_rows",
                "proxy_rows",
                "created_at",
            ),
            timestamp_column="created_at",
            freshness_seconds=24 * 60 * 60,
            quality_rules=("schema_required_columns", "freshness"),
        ),
        DatasetSpec(
            dataset_id="alpha_evidence",
            layer="gold",
            relative_path=Path("gold") / "alpha_evidence",
            owner="research",
            producer="qlab build-alpha-evidence",
            consumers=("gate", "web", "expert-export"),
            primary_key=("alpha_id", "symbol", "horizon_hours", "created_at"),
            required_columns=(
                "alpha_id",
                "status",
                "evidence_status",
                "sample_count",
                "created_at",
            ),
            timestamp_column="created_at",
            utc_timestamp_columns=("created_at",),
            freshness_seconds=24 * 60 * 60,
            quality_rules=("schema_required_columns", "freshness"),
        ),
        DatasetSpec(
            dataset_id="gate_decision",
            layer="gold",
            relative_path=Path("gold") / "gate_decision",
            owner="research",
            producer="qlab publish-gate-decisions",
            consumers=("risk", "web", "expert-export"),
            required_columns=("alpha_id", "status", "decision", "created_at"),
            timestamp_column="created_at",
            utc_timestamp_columns=("created_at",),
            freshness_seconds=6 * 60 * 60,
            quality_rules=("schema_required_columns", "freshness"),
        ),
        DatasetSpec(
            dataset_id="risk_permission",
            layer="gold",
            relative_path=Path("gold") / "risk_permission",
            owner="risk",
            producer="qlab publish-risk-permission",
            consumers=("api", "v5", "web", "expert-export"),
            required_columns=(
                "permission",
                "permission_status",
                "as_of_ts",
                "expires_at",
                "enforceable",
            ),
            timestamp_column="as_of_ts",
            utc_timestamp_columns=("as_of_ts", "expires_at"),
            freshness_seconds=90 * 60,
            quality_rules=(
                "schema_required_columns",
                "utc_timestamps",
                "risk_permission_freshness",
            ),
        ),
        DatasetSpec(
            dataset_id="strategy_opportunity_advisory",
            layer="gold",
            relative_path=Path("gold") / "strategy_opportunity_advisory",
            owner="research",
            producer="qlab export-daily",
            consumers=("api", "v5", "web", "expert-export"),
            required_columns=(
                "strategy_candidate",
                "symbol",
                "decision",
                "recommended_mode",
                "max_live_notional_usdt",
                "generated_at",
                "contract_version",
            ),
            timestamp_column="generated_at",
            utc_timestamp_columns=("generated_at",),
            freshness_seconds=3 * 60 * 60,
            quality_rules=("schema_required_columns", "freshness"),
        ),
        DatasetSpec(
            dataset_id="v5_candidate_event",
            layer="silver",
            relative_path=Path("silver") / "v5_candidate_event",
            owner="v5-telemetry",
            producer="qlab ingest-v5-bundle",
            consumers=("research", "web", "expert-export"),
            primary_key=("candidate_id", "run_id", "ts_utc", "symbol"),
            required_columns=("candidate_id", "run_id", "ts_utc", "symbol", "final_decision"),
            timestamp_column="ts_utc",
            utc_timestamp_columns=("ts_utc",),
            freshness_seconds=3 * 60 * 60,
            quality_rules=("schema_required_columns", "primary_key_unique", "freshness"),
        ),
        DatasetSpec(
            dataset_id="v5_quant_lab_usage",
            layer="silver",
            relative_path=Path("silver") / "v5_quant_lab_usage",
            owner="v5-telemetry",
            producer="qlab ingest-v5-bundle",
            consumers=("readiness", "web", "expert-export"),
            required=False,
            required_columns=("ts_utc", "run_id", "mode"),
            timestamp_column="ts_utc",
            utc_timestamp_columns=("ts_utc",),
            freshness_seconds=3 * 60 * 60,
            quality_rules=("schema_required_columns", "freshness"),
        ),
        DatasetSpec(
            dataset_id="strategy_health_daily",
            layer="gold",
            relative_path=Path("gold") / "strategy_health_daily",
            owner="v5-telemetry",
            producer="qlab analyze-v5-telemetry",
            consumers=("readiness", "web", "expert-export"),
            required_columns=("day", "status", "latest_bundle_ts", "created_at"),
            timestamp_column="created_at",
            utc_timestamp_columns=("created_at", "latest_bundle_ts"),
            freshness_seconds=3 * 60 * 60,
            quality_rules=("schema_required_columns", "freshness", "telemetry_dedupe_health"),
        ),
        DatasetSpec(
            dataset_id="api_request_metrics",
            layer="bronze",
            relative_path=Path("bronze") / "api_request_metrics",
            owner="ops",
            producer="FastAPI middleware",
            consumers=("ops-summary", "expert-export"),
            write_mode="append",
            required=False,
            required_columns=("request_ts", "method", "path", "status_code", "duration_ms"),
            timestamp_column="request_ts",
            utc_timestamp_columns=("request_ts",),
            retention_days=30,
            quality_rules=("schema_required_columns", "utc_timestamps"),
        ),
        DatasetSpec(
            dataset_id="job_run_history",
            layer="gold",
            relative_path=Path("gold") / "job_run_history",
            owner="ops",
            producer="run_with_job_metrics",
            consumers=("ops-summary", "expert-export"),
            write_mode="upsert",
            required=False,
            required_columns=("job_name", "status", "started_at", "finished_at", "duration_s"),
            timestamp_column="finished_at",
            utc_timestamp_columns=("started_at", "finished_at"),
            retention_days=90,
            quality_rules=("schema_required_columns", "utc_timestamps"),
        ),
        DatasetSpec(
            dataset_id="lake_file_health_daily",
            layer="gold",
            relative_path=Path("gold") / "lake_file_health_daily",
            owner="ops",
            producer="qlab lake-health",
            consumers=("ops-summary", "expert-export"),
            required=False,
            required_columns=("day", "dataset", "parquet_file_count", "status", "created_at"),
            timestamp_column="created_at",
            utc_timestamp_columns=("created_at",),
            freshness_seconds=24 * 60 * 60,
            quality_rules=("schema_required_columns", "freshness"),
        ),
        DatasetSpec(
            dataset_id="trade_print",
            layer="silver",
            relative_path=Path("silver") / "trade_print",
            owner="market-data",
            producer="OKX public WebSocket",
            consumers=("costs", "web", "expert-export"),
            write_mode="append",
            required=False,
            timestamp_column="ts",
            freshness_seconds=30 * 60,
            quality_rules=("freshness",),
        ),
        DatasetSpec(
            dataset_id="orderbook_snapshot",
            layer="silver",
            relative_path=Path("silver") / "orderbook_snapshot",
            owner="market-data",
            producer="OKX public WebSocket",
            consumers=("costs", "web", "expert-export"),
            write_mode="append",
            required=False,
            timestamp_column="ts",
            freshness_seconds=30 * 60,
            quality_rules=("freshness",),
        ),
        DatasetSpec(
            dataset_id="okx_public_ws",
            layer="bronze",
            relative_path=Path("bronze") / "okx_public_ws",
            owner="market-data",
            producer="OKX public WebSocket",
            consumers=("normalization", "web", "expert-export"),
            write_mode="append",
            required=False,
            timestamp_column="ingest_ts",
            freshness_seconds=30 * 60,
            quality_rules=("freshness",),
        ),
    ]
    return {spec.dataset_id: spec for spec in specs}


def dataset_registry() -> dict[str, DatasetSpec]:
    return core_dataset_specs()


def get_dataset_spec(dataset_id: str) -> DatasetSpec | None:
    return dataset_registry().get(dataset_id)


def dataset_names() -> list[str]:
    return sorted(dataset_registry())


def dataset_path_map() -> dict[str, Path]:
    return {name: spec.relative_path for name, spec in dataset_registry().items()}


def dataset_registry_rows() -> list[dict[str, Any]]:
    return [spec.to_row() for spec in dataset_registry().values()]
