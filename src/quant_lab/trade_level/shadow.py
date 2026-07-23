from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import (
    read_parquet_dataset,
    write_parquet_dataset,
    write_snapshot_meta,
)
from quant_lab.export_plane.status import atomic_write_json
from quant_lab.research.candidate_labels import CANDIDATE_LABEL_DATASET
from quant_lab.research_plane.signatures import model_content_sha256
from quant_lab.research_plane.trade_level_history_publish import (
    TRADE_LEVEL_HISTORY_GENERATION_POINTER,
    verify_trade_level_history_generation_fast,
)
from quant_lab.trade_level.judgment import (
    RISK_PERMISSION_DATASET,
    TRADE_LEVEL_BUCKET_POLICY_DATASET,
    TRADE_LEVEL_CONTROL_OUTPUT_SPECS,
    TRADE_LEVEL_JUDGMENT_DATASET,
    TRADE_LEVEL_SIMILARITY_DATASET,
    TRADE_OPPORTUNITY_EVENT_DATASET,
    TRADE_OPPORTUNITY_LABEL_DATASET,
    V5_CANDIDATE_EVENT_DATASET,
    V5_ORDER_LIFECYCLE_DATASET,
    V5_ROUNDTRIP_DATASET,
    V5_TRADE_EVENT_DATASET,
    _schema_version_for_dataset,
    build_trade_level_control_frames,
    build_trade_opportunity_events,
)
from quant_lab.trade_level.labels import (
    build_trade_opportunity_labels,
)
from quant_lab.trade_level.similarity import (
    build_trade_level_similarity_outcome,
)

_SIMILARITY_METRICS = (
    "similar_sample_count",
    "similar_mean_after_cost_bps",
    "similar_median_after_cost_bps",
    "similar_p25_after_cost_bps",
    "similar_hit_rate",
    "similar_max_adverse_bps",
    "recent_7d_similar_mean",
)


class TradeLevelLegacyShadowResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    history_generation_id: str
    history_candidate_binding_status: str
    status: str
    report_path: str
    legacy_event_rows: int = Field(ge=0)
    accepted_event_rows: int = Field(ge=0)
    similarity_mismatch_count: int = Field(ge=0)
    downstream_decision_mismatch_count: int = Field(ge=0)
    new_micro_canary_allow_count: int = Field(ge=0)
    published_new_micro_canary_allow_count: int = Field(ge=0)
    risk_permission_unchanged: bool
    cloud_duration_seconds: float = Field(ge=0)
    peak_rss_bytes: int = Field(ge=0)
    live_order_effect: str


def build_and_publish_trade_level_legacy_control_shadow(
    lake_root: str | Path,
    report_root: str | Path,
    *,
    as_of_date: str | date | None = None,
) -> TradeLevelLegacyShadowResult:
    """Keep legacy cloud controls active without overwriting NAS history Gold."""

    started = time.perf_counter()
    root = Path(lake_root).resolve(strict=True)
    reports = Path(report_root)
    reports.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC)
    day = _parse_day(as_of_date, generated_at)
    pointer = _load_history_pointer(root)
    generation_id = str(pointer.get("generation_id") or "")
    if not generation_id:
        raise RuntimeError(
            "trade_level_shadow_history_generation_missing"
        )
    history_candidate_binding_status = "CURRENT"
    try:
        verify_trade_level_history_generation_fast(
            root,
            generation_id,
            expected_input_fingerprint=str(
                pointer.get("input_fingerprint_digest") or ""
            ),
            expected_candidate_generation_id=str(
                pointer.get("candidate_evidence_generation_id") or ""
            ),
            expected_candidate_generation_digest=str(
                pointer.get("candidate_evidence_generation_digest") or ""
            ),
        )
    except RuntimeError as exc:
        if str(exc) != (
            "trade_level_history_generation_candidate_binding_mismatch"
        ):
            raise
        # Candidate Evidence advances independently while the Trade-Level
        # request timer remains intentionally disabled during Shadow. Keep
        # validating the immutable history generation itself, then compare it
        # diagnostically while the legacy cloud path remains authoritative.
        verify_trade_level_history_generation_fast(
            root,
            generation_id,
            expected_input_fingerprint=str(
                pointer.get("input_fingerprint_digest") or ""
            ),
            expected_candidate_generation_id=str(
                pointer.get("candidate_evidence_generation_id") or ""
            ),
            expected_candidate_generation_digest=str(
                pointer.get("candidate_evidence_generation_digest") or ""
            ),
            require_current_candidate_binding=False,
        )
        history_candidate_binding_status = (
            "STALE_CANDIDATE_GENERATION"
        )

    risk_permissions = read_parquet_dataset(
        root / RISK_PERMISSION_DATASET
    )
    risk_digest_before = _frame_digest(
        "risk_permission",
        risk_permissions,
    )
    candidate_events = read_parquet_dataset(
        root / V5_CANDIDATE_EVENT_DATASET
    )
    candidate_labels = read_parquet_dataset(
        root / CANDIDATE_LABEL_DATASET
    )
    v5_trades = read_parquet_dataset(root / V5_TRADE_EVENT_DATASET)
    v5_roundtrips = read_parquet_dataset(
        root / V5_ROUNDTRIP_DATASET
    )
    order_lifecycles = read_parquet_dataset(
        root / V5_ORDER_LIFECYCLE_DATASET
    )
    legacy_events = build_trade_opportunity_events(
        candidate_events,
        risk_permissions=risk_permissions,
        v5_trades=v5_trades,
        order_lifecycles=order_lifecycles,
        created_at=generated_at,
    )
    legacy_labels = build_trade_opportunity_labels(
        legacy_events,
        candidate_labels,
        created_at=generated_at,
    )
    legacy_similarity = build_trade_level_similarity_outcome(
        legacy_events,
        legacy_labels,
        created_at=generated_at,
    )

    current_judgments = read_parquet_dataset(
        root / TRADE_LEVEL_JUDGMENT_DATASET
    )
    current_policy = read_parquet_dataset(
        root / TRADE_LEVEL_BUCKET_POLICY_DATASET
    )
    allowed_event_ids = frozenset(
        str(value)
        for value in _filtered_values(
            current_judgments,
            key="event_id",
            filter_column="trade_level_decision",
            filter_value="MICRO_CANARY_ALLOW",
        )
    )
    allowed_bucket_keys = frozenset(
        str(value)
        for value in _filtered_values(
            current_policy,
            key="bucket_key",
            filter_column="policy_action",
            filter_value="MICRO_CANARY_ALLOW",
        )
    )
    legacy_control = build_trade_level_control_frames(
        events=legacy_events,
        labels=legacy_labels,
        similarity=legacy_similarity,
        v5_trades=v5_trades,
        v5_roundtrips=v5_roundtrips,
        order_lifecycles=order_lifecycles,
        as_of_date=day,
        created_at=generated_at,
        shadow_allowed_allow_event_ids=allowed_event_ids,
        shadow_allowed_allow_bucket_keys=allowed_bucket_keys,
    )

    accepted_events = read_parquet_dataset(
        root / TRADE_OPPORTUNITY_EVENT_DATASET
    )
    accepted_labels = read_parquet_dataset(
        root / TRADE_OPPORTUNITY_LABEL_DATASET
    )
    accepted_similarity = read_parquet_dataset(
        root / TRADE_LEVEL_SIMILARITY_DATASET
    )
    accepted_control = build_trade_level_control_frames(
        events=accepted_events,
        labels=accepted_labels,
        similarity=accepted_similarity,
        v5_trades=v5_trades,
        v5_roundtrips=v5_roundtrips,
        order_lifecycles=order_lifecycles,
        as_of_date=day,
        created_at=generated_at,
    )

    similarity_diffs = _metric_diff_counts(
        legacy_similarity,
        accepted_similarity,
        key="event_id",
        metrics=_SIMILARITY_METRICS,
    )
    decision_diffs = _metric_diff_counts(
        legacy_control["trade_level_judgment"],
        accepted_control["trade_level_judgment"],
        key="event_id",
        metrics=(
            "trade_level_decision",
            "max_single_order_usdt",
            "daily_trade_limit",
        ),
    )
    current_allow_ids = set(allowed_event_ids)
    accepted_allow_ids = set(
        str(value)
        for value in _filtered_values(
            accepted_control["trade_level_judgment"],
            key="event_id",
            filter_column="trade_level_decision",
            filter_value="MICRO_CANARY_ALLOW",
        )
    )
    new_allow_ids = sorted(accepted_allow_ids - current_allow_ids)
    published_judgments = legacy_control["trade_level_judgment"]
    published_policy = legacy_control["trade_level_bucket_policy"]
    published_allow_ids = set(
        str(value)
        for value in _filtered_values(
            published_judgments,
            key="event_id",
            filter_column="trade_level_decision",
            filter_value="MICRO_CANARY_ALLOW",
        )
    )
    published_new_allow_ids = sorted(
        published_allow_ids - current_allow_ids
    )
    judgment_limit_increases = _limit_increase_keys(
        current_judgments,
        published_judgments,
        key="event_id",
    )
    policy_limit_increases = _limit_increase_keys(
        current_policy,
        published_policy,
        key="bucket_key",
    )
    if published_new_allow_ids:
        raise RuntimeError(
            "trade_level_shadow_new_micro_canary_allow_forbidden"
        )
    if judgment_limit_increases or policy_limit_increases:
        raise RuntimeError(
            "trade_level_shadow_order_limit_increase_forbidden:"
            + ",".join(
                [
                    *(
                        f"event:{value}"
                        for value in judgment_limit_increases
                    ),
                    *(
                        f"bucket:{value}"
                        for value in policy_limit_increases
                    ),
                ]
            )
        )
    risk_before_publish = read_parquet_dataset(
        root / RISK_PERMISSION_DATASET
    )
    if _frame_digest("risk_permission", risk_before_publish) != (
        risk_digest_before
    ):
        raise RuntimeError(
            "trade_level_shadow_risk_permission_changed"
        )

    for dataset_name, relative_path in TRADE_LEVEL_CONTROL_OUTPUT_SPECS:
        frame = legacy_control[dataset_name]
        dataset_path = root / relative_path
        write_parquet_dataset(frame, dataset_path)
        write_snapshot_meta(
            dataset_path,
            dataset_name=dataset_name,
            frame=frame,
            schema_version=_schema_version_for_dataset(dataset_name),
            generated_at=generated_at,
        )

    published_max_order = _max_numeric(
        published_judgments,
        "max_single_order_usdt",
    )
    published_max_daily_limit = _max_numeric(
        published_judgments,
        "daily_trade_limit",
    )

    risk_after = read_parquet_dataset(root / RISK_PERMISSION_DATASET)
    risk_digest_after = _frame_digest("risk_permission", risk_after)
    risk_unchanged = risk_digest_before == risk_digest_after
    if not risk_unchanged:
        raise RuntimeError(
            "trade_level_shadow_risk_permission_changed"
        )

    legacy_decisions = _decision_counts(
        legacy_control["trade_level_judgment"]
    )
    accepted_decisions = _decision_counts(
        accepted_control["trade_level_judgment"]
    )
    report = {
        "schema_version": "trade_level_history_shadow_report.v1",
        "shadow_mode": "legacy_cloud_control_with_nas_history_compare",
        "as_of_date": day.isoformat(),
        "history_generation_id": generation_id,
        "history_generation_digest": pointer.get(
            "generation_digest"
        ),
        "history_input_fingerprint": pointer.get(
            "input_fingerprint_digest"
        ),
        "history_candidate_binding_status": (
            history_candidate_binding_status
        ),
        "history_candidate_generation_id": pointer.get(
            "candidate_evidence_generation_id"
        ),
        "legacy_history": {
            "event_rows": legacy_events.height,
            "label_rows": legacy_labels.height,
            "similarity_rows": legacy_similarity.height,
        },
        "accepted_nas_history": {
            "event_rows": accepted_events.height,
            "label_rows": accepted_labels.height,
            "similarity_rows": accepted_similarity.height,
        },
        "similarity_diff_counts": similarity_diffs,
        "future_label_removed_candidate_count": int(
            similarity_diffs.get("similar_sample_count", 0)
        ),
        "downstream_decision_diff_counts": decision_diffs,
        "legacy_decision_counts": legacy_decisions,
        "accepted_decision_counts": accepted_decisions,
        "new_micro_canary_allow_event_ids": new_allow_ids,
        "published_new_micro_canary_allow_event_ids": (
            published_new_allow_ids
        ),
        "published_max_single_order_usdt": published_max_order,
        "published_max_daily_trade_limit": (
            published_max_daily_limit
        ),
        "judgment_order_limit_increase_event_ids": (
            judgment_limit_increases
        ),
        "policy_order_limit_increase_bucket_keys": (
            policy_limit_increases
        ),
        "risk_permission_digest_before": risk_digest_before,
        "risk_permission_digest_after": risk_digest_after,
        "risk_permission_unchanged": risk_unchanged,
        "history_gold_written_by_legacy_path": False,
        "control_gold_source": "legacy_cloud_shadow",
        "automatic_promotion": False,
        "live_order_effect": "no_new_allow_shadow",
        "generated_at": generated_at.isoformat(),
        "cloud_duration_seconds": time.perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    token = generated_at.strftime("%Y%m%dT%H%M%SZ")
    report_path = reports / f"{token}-{generation_id}.json"
    atomic_write_json(report_path, report)
    atomic_write_json(reports / "latest.json", report)
    return TradeLevelLegacyShadowResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        history_generation_id=generation_id,
        history_candidate_binding_status=(
            history_candidate_binding_status
        ),
        status="PASS",
        report_path=str(report_path),
        legacy_event_rows=legacy_events.height,
        accepted_event_rows=accepted_events.height,
        similarity_mismatch_count=sum(similarity_diffs.values()),
        downstream_decision_mismatch_count=sum(
            decision_diffs.values()
        ),
        new_micro_canary_allow_count=len(new_allow_ids),
        published_new_micro_canary_allow_count=len(
            published_new_allow_ids
        ),
        risk_permission_unchanged=risk_unchanged,
        cloud_duration_seconds=float(
            report["cloud_duration_seconds"]
        ),
        peak_rss_bytes=int(report["peak_rss_bytes"]),
        live_order_effect="no_new_allow_shadow",
    )


def _load_history_pointer(root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            (root / TRADE_LEVEL_HISTORY_GENERATION_POINTER).read_text(
                "utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "trade_level_shadow_history_pointer_invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            "trade_level_shadow_history_pointer_invalid"
        )
    return payload


def _metric_diff_counts(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    key: str,
    metrics: tuple[str, ...],
) -> dict[str, int]:
    if key not in left.columns or key not in right.columns:
        return {metric: max(left.height, right.height, 1) for metric in metrics}
    left_columns = [key, *(metric for metric in metrics if metric in left.columns)]
    right_columns = [key, *(metric for metric in metrics if metric in right.columns)]
    joined = left.select(left_columns).join(
        right.select(right_columns),
        on=key,
        how="full",
        suffix="_accepted",
        coalesce=True,
    )
    result: dict[str, int] = {}
    for metric in metrics:
        accepted = f"{metric}_accepted"
        if metric not in joined.columns or accepted not in joined.columns:
            result[metric] = max(joined.height, 1)
            continue
        dtype = joined.schema[metric]
        if dtype.is_float():
            mismatch = ~(
                (pl.col(metric).is_null() & pl.col(accepted).is_null())
                | (
                    pl.col(metric).is_not_null()
                    & pl.col(accepted).is_not_null()
                    & (
                        (pl.col(metric) - pl.col(accepted)).abs()
                        <= 1e-9
                    )
                )
            )
        else:
            mismatch = ~pl.col(metric).eq_missing(pl.col(accepted))
        result[metric] = joined.filter(mismatch).height
    return result


def _filtered_values(
    frame: pl.DataFrame,
    *,
    key: str,
    filter_column: str,
    filter_value: str,
) -> list[object]:
    if (
        frame.is_empty()
        or key not in frame.columns
        or filter_column not in frame.columns
    ):
        return []
    return (
        frame.filter(pl.col(filter_column) == filter_value)
        .get_column(key)
        .drop_nulls()
        .to_list()
    )


def _decision_counts(frame: pl.DataFrame) -> dict[str, int]:
    if frame.is_empty() or "trade_level_decision" not in frame.columns:
        return {}
    grouped = frame.group_by("trade_level_decision").len()
    return {
        str(row["trade_level_decision"]): int(row["len"])
        for row in grouped.to_dicts()
    }


def _max_numeric(frame: pl.DataFrame, column: str) -> float:
    if frame.is_empty() or column not in frame.columns:
        return 0.0
    value = frame.get_column(column).fill_null(0.0).max()
    return float(value or 0.0)


def _limit_increase_keys(
    current: pl.DataFrame,
    proposed: pl.DataFrame,
    *,
    key: str,
) -> list[str]:
    limit_columns = (
        "max_single_order_usdt",
        "daily_trade_limit",
    )
    if key not in proposed.columns:
        return ["missing_proposed_key"]
    current_limits: dict[str, tuple[float, float]] = {}
    if key in current.columns:
        for row in current.to_dicts():
            value = str(row.get(key) or "")
            if not value:
                continue
            current_limits[value] = tuple(
                float(row.get(column) or 0.0)
                for column in limit_columns
            )
    increases: list[str] = []
    for row in proposed.to_dicts():
        value = str(row.get(key) or "")
        if not value:
            increases.append("missing_key")
            continue
        current_order, current_daily = current_limits.get(
            value,
            (0.0, 0.0),
        )
        proposed_order = float(
            row.get("max_single_order_usdt") or 0.0
        )
        proposed_daily = float(row.get("daily_trade_limit") or 0.0)
        if (
            proposed_order > current_order + 1e-9
            or proposed_daily > current_daily
        ):
            increases.append(value)
    return sorted(set(increases))


def _frame_digest(label: str, frame: pl.DataFrame) -> str:
    normalized = frame
    if frame.columns:
        sort_columns = [
            column
            for column in (
                "as_of_ts",
                "decision_ts",
                "event_id",
                "symbol",
            )
            if column in frame.columns
        ]
        if sort_columns:
            normalized = frame.sort(sort_columns)
    return model_content_sha256(
        {
            "schema_version": "trade_level_shadow_frame_digest.v1",
            "label": label,
            "columns": list(normalized.columns),
            "rows": normalized.to_dicts(),
        }
    )


def _parse_day(
    value: str | date | None,
    generated_at: datetime,
) -> date:
    if isinstance(value, date):
        return value
    if value and str(value).lower() != "auto":
        return date.fromisoformat(str(value))
    return generated_at.date()


def _peak_rss_bytes() -> int:
    try:
        import resource  # noqa: PLC0415

        return int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ) * 1024
    except (ImportError, AttributeError):
        try:
            import psutil  # type: ignore[import-untyped]  # noqa: PLC0415

            return int(psutil.Process().memory_info().rss)
        except (ImportError, OSError):
            return 0
