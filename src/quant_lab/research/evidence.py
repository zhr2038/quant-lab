import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab import __version__
from quant_lab.contracts.models import AlphaEvidence, AlphaResearchSpec
from quant_lab.data.lake import read_parquet_dataset
from quant_lab.research.backtest import simulate_long_only_oos
from quant_lab.research.ic import compute_ic, compute_rank_ic
from quant_lab.research.labels import build_forward_return_labels, validate_no_label_lookahead

FEATURE_VALUE_DATASET = Path("gold") / "feature_value"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"
DEFAULT_RESEARCH_COST_BPS = 25.0


class AlphaDatasetResult:
    def __init__(self, frame: pl.DataFrame, warnings: list[str]) -> None:
        self.frame = frame
        self.warnings = warnings


class AlphaEvidenceBuildResult:
    def __init__(
        self,
        evidence: AlphaEvidence | None,
        dataset: pl.DataFrame,
        warnings: list[str],
        status: str,
    ) -> None:
        self.evidence = evidence
        self.dataset = dataset
        self.warnings = warnings
        self.status = status


def build_alpha_dataset(lake_root: str | Path, spec: AlphaResearchSpec) -> AlphaDatasetResult:
    root = Path(lake_root)
    warnings: list[str] = []
    features = _load_features(root, spec, warnings)
    if features.is_empty():
        warnings.append("feature_value missing or no rows matched alpha spec")
        return AlphaDatasetResult(_empty_alpha_dataset(spec.feature_names), _dedupe(warnings))

    market_bars = _load_market_bars(root, spec)
    if market_bars.is_empty():
        warnings.append("market_bar missing or no rows matched alpha spec")
        return AlphaDatasetResult(_empty_alpha_dataset(spec.feature_names), _dedupe(warnings))

    labels = build_forward_return_labels(
        market_bars,
        horizon_bars=spec.label_horizon_bars,
        decision_delay_bars=spec.decision_delay_bars,
    )
    validate_no_label_lookahead(labels)
    if labels.is_empty():
        warnings.append("insufficient market_bar history for forward labels")
        return AlphaDatasetResult(_empty_alpha_dataset(spec.feature_names), _dedupe(warnings))

    feature_wide = _pivot_feature_values(features, spec, warnings)
    dataset = labels.join(feature_wide, on=["symbol", "timeframe", "feature_ts"], how="inner")
    if dataset.is_empty():
        warnings.append("feature_value and labels did not overlap")
        return AlphaDatasetResult(_empty_alpha_dataset(spec.feature_names), _dedupe(warnings))

    dataset = _add_alpha_score(dataset, spec.feature_names)
    dataset = _attach_costs(dataset, root, spec, warnings)
    return AlphaDatasetResult(dataset.sort(["decision_ts", "symbol"]), _dedupe(warnings))


def build_alpha_evidence(
    lake_root: str | Path,
    spec: AlphaResearchSpec,
) -> AlphaEvidenceBuildResult:
    dataset_result = build_alpha_dataset(lake_root, spec)
    dataset = dataset_result.frame
    warnings = list(dataset_result.warnings)
    now = datetime.now(UTC)
    if dataset.is_empty():
        return AlphaEvidenceBuildResult(None, dataset, warnings, "insufficient_data")

    total_rows = dataset.height
    valid = dataset.filter(
        pl.col("forward_return").is_not_null() & pl.col("alpha_score").is_not_null()
    )
    valid_rows = valid.height
    coverage = valid_rows / total_rows if total_rows else 0.0
    status = "ok"
    if valid_rows < spec.min_samples:
        warnings.append("insufficient_samples")
        status = "insufficient_samples"

    ic_stats = compute_ic(valid)
    rank_ic_stats = compute_rank_ic(valid)
    if ic_stats.status != "ok":
        warnings.append(ic_stats.status)
    if rank_ic_stats.status != "ok":
        warnings.append(f"rank_{rank_ic_stats.status}")

    oos_stats = simulate_long_only_oos(
        valid,
        top_quantile=spec.top_quantile,
        cost_quantile=spec.cost_quantile,
    )
    warnings.extend(oos_stats.warnings)

    start_ts = _frame_datetime(dataset, "feature_ts", "min") or now
    end_ts = _frame_datetime(dataset, "feature_ts", "max") or (start_ts + timedelta(seconds=1))
    if end_ts <= start_ts:
        end_ts = start_ts + timedelta(seconds=1)
    if status == "ok" and _evidence_window_is_stale(end_ts, now):
        warnings.append("stale")
        status = "stale"

    evidence = AlphaEvidence(
        alpha_id=spec.alpha_id,
        version=spec.version,
        data_version=_data_version(dataset),
        feature_version=_feature_version(spec),
        cost_model_version=_cost_model_version(dataset),
        universe_id=spec.universe_id,
        start_ts=start_ts,
        end_ts=end_ts,
        coverage=coverage,
        ic_mean=ic_stats.mean,
        ic_tstat=ic_stats.tstat,
        rank_ic_mean=rank_ic_stats.mean,
        rank_ic_tstat=rank_ic_stats.tstat,
        edge_cost_ratio=oos_stats.edge_cost_ratio,
        oos_sharpe=oos_stats.oos_sharpe,
        oos_sortino=oos_stats.oos_sortino,
        oos_cagr=oos_stats.oos_cagr,
        oos_max_drawdown=oos_stats.oos_max_drawdown,
        profit_factor=oos_stats.profit_factor,
        turnover=oos_stats.turnover,
        cost_ratio=oos_stats.cost_ratio,
        profitable_folds_ratio=oos_stats.profitable_folds_ratio,
        train_oos_decay=oos_stats.train_oos_decay,
        pbo_score=None,
        paper_days=0,
        paper_slippage_coverage=0.0,
        created_at=now,
        evidence_status=status,
    )
    return AlphaEvidenceBuildResult(evidence, dataset, _dedupe(warnings), status)


def _evidence_window_is_stale(end_ts: datetime, now: datetime) -> bool:
    value = os.environ.get("QUANT_LAB_ALPHA_EVIDENCE_STALE_DAYS", "7")
    try:
        stale_days = int(value)
    except ValueError:
        stale_days = 7
    if stale_days <= 0:
        return False
    return end_ts < now - timedelta(days=stale_days)


def _load_features(root: Path, spec: AlphaResearchSpec, warnings: list[str]) -> pl.DataFrame:
    df = read_parquet_dataset(root / FEATURE_VALUE_DATASET)
    if df.is_empty():
        return df
    required = {"feature_set", "feature_version", "feature_name", "timeframe", "symbol", "ts"}
    missing = sorted(required.difference(df.columns))
    if missing:
        warnings.append(f"feature_value missing columns: {','.join(missing)}")
        return pl.DataFrame()
    filtered = _normalize_datetime(df, "ts").filter(
        (pl.col("feature_set") == spec.feature_set)
        & (pl.col("feature_version") == spec.feature_version)
        & (pl.col("feature_name").is_in(spec.feature_names))
        & (pl.col("timeframe") == spec.timeframe)
    )
    if spec.start is not None:
        filtered = filtered.filter(pl.col("ts") >= spec.start)
    if spec.end is not None:
        filtered = filtered.filter(pl.col("ts") <= spec.end)
    return filtered


def _load_market_bars(root: Path, spec: AlphaResearchSpec) -> pl.DataFrame:
    df = read_parquet_dataset(root / MARKET_BAR_DATASET)
    if df.is_empty() or "ts" not in df.columns:
        return pl.DataFrame()
    normalized = _normalize_datetime(df, "ts")
    if "is_closed" in normalized.columns:
        normalized = normalized.filter(pl.col("is_closed"))
    filtered = normalized.filter(pl.col("timeframe") == spec.timeframe)
    if spec.start is not None:
        filtered = filtered.filter(pl.col("ts") >= spec.start)
    if spec.end is not None:
        end_with_label = spec.end + _bar_delta(spec.timeframe) * (
            spec.decision_delay_bars + spec.label_horizon_bars
        )
        filtered = filtered.filter(pl.col("ts") <= end_with_label)
    return filtered


def _pivot_feature_values(
    features: pl.DataFrame,
    spec: AlphaResearchSpec,
    warnings: list[str],
) -> pl.DataFrame:
    key_columns = ["feature_set", "feature_name", "feature_version", "symbol", "timeframe", "ts"]
    duplicate_count = (
        features.group_by(key_columns)
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        warnings.append(f"duplicate_feature_keys:{duplicate_count}")
    selected = (
        features.select(["symbol", "timeframe", "ts", "feature_name", "value"])
        .unique(subset=["symbol", "timeframe", "ts", "feature_name"], keep="last")
        .rename({"ts": "feature_ts"})
    )
    wide = selected.pivot(
        on="feature_name",
        index=["symbol", "timeframe", "feature_ts"],
        values="value",
        aggregate_function="last",
    )
    for name in spec.feature_names:
        if name not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    return wide


def _add_alpha_score(dataset: pl.DataFrame, feature_names: list[str]) -> pl.DataFrame:
    expressions = [pl.col(name).cast(pl.Float64, strict=False) for name in feature_names]
    return dataset.with_columns(
        [
            pl.mean_horizontal(expressions).alias("alpha_score"),
            pl.sum_horizontal([expr.is_not_null().cast(pl.Int64) for expr in expressions]).alias(
                "feature_non_null_count"
            ),
            pl.lit(len(feature_names)).alias("feature_required_count"),
        ]
    )


def _attach_costs(
    dataset: pl.DataFrame,
    root: Path,
    spec: AlphaResearchSpec,
    warnings: list[str],
) -> pl.DataFrame:
    costs = read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET)
    cost_column = f"total_cost_bps_{spec.cost_quantile}"
    if costs.is_empty() or cost_column not in costs.columns:
        warnings.append("cost_bucket_daily missing; using research global default cost")
        return dataset.with_columns(
            [
                pl.lit(DEFAULT_RESEARCH_COST_BPS).alias(cost_column),
                pl.lit("global_default_v0").alias("cost_model_version"),
                pl.lit("global_default").alias("cost_source"),
            ]
        )

    latest = costs
    if "day" in latest.columns:
        latest = latest.sort("day")
    latest = latest.unique(subset=["symbol"], keep="last")
    join_cols = ["symbol", cost_column]
    if "cost_model_version" in latest.columns:
        join_cols.append("cost_model_version")
    if "source" in latest.columns:
        latest = latest.rename({"source": "cost_source"})
        join_cols.append("cost_source")
    joined = dataset.join(latest.select(join_cols), on="symbol", how="left")
    if cost_column not in joined.columns:
        joined = joined.with_columns(pl.lit(DEFAULT_RESEARCH_COST_BPS).alias(cost_column))
    joined = joined.with_columns(
        [
            pl.col(cost_column).fill_null(DEFAULT_RESEARCH_COST_BPS),
            (
                pl.col("cost_model_version").fill_null("global_default_v0")
                if "cost_model_version" in joined.columns
                else pl.lit("global_default_v0").alias("cost_model_version")
            ),
            (
                pl.col("cost_source").fill_null("global_default")
                if "cost_source" in joined.columns
                else pl.lit("global_default").alias("cost_source")
            ),
        ]
    )
    return joined


def _empty_alpha_dataset(feature_names: list[str]) -> pl.DataFrame:
    schema: dict[str, Any] = {
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "feature_ts": pl.Datetime(time_zone="UTC"),
        "decision_ts": pl.Datetime(time_zone="UTC"),
        "label_ts": pl.Datetime(time_zone="UTC"),
        "forward_return": pl.Float64,
        "alpha_score": pl.Float64,
    }
    for name in feature_names:
        schema[name] = pl.Float64
    return pl.DataFrame(schema=schema)


def _normalize_datetime(df: pl.DataFrame, column: str) -> pl.DataFrame:
    if df.schema.get(column) == pl.String:
        return df.with_columns(pl.col(column).str.to_datetime(time_zone="UTC", strict=False))
    return df.with_columns(pl.col(column).cast(pl.Datetime(time_zone="UTC")).alias(column))


def _frame_datetime(df: pl.DataFrame, column: str, op: str) -> datetime | None:
    if df.is_empty() or column not in df.columns:
        return None
    value = df.select(getattr(pl.col(column), op)()).item()
    return value.astimezone(UTC) if isinstance(value, datetime) else None


def _data_version(dataset: pl.DataFrame) -> str:
    latest = _frame_datetime(dataset, "label_ts", "max")
    rendered = latest.isoformat() if latest else "none"
    return f"market_bar:{rendered}:{dataset.height}"


def _feature_version(spec: AlphaResearchSpec) -> str:
    return f"{spec.feature_set}:{spec.feature_version}:{','.join(spec.feature_names)}"


def _cost_model_version(dataset: pl.DataFrame) -> str:
    if dataset.is_empty() or "cost_model_version" not in dataset.columns:
        return "global_default_v0"
    values = sorted({str(value) for value in dataset["cost_model_version"].drop_nulls().to_list()})
    return "+".join(values) if values else "global_default_v0"


def _bar_delta(timeframe: str) -> timedelta:
    normalized = timeframe.strip().lower()
    if normalized.endswith("h"):
        return timedelta(hours=int(normalized[:-1] or "1"))
    if normalized.endswith("m"):
        return timedelta(minutes=int(normalized[:-1] or "1"))
    if normalized.endswith("d"):
        return timedelta(days=int(normalized[:-1] or "1"))
    return timedelta(hours=1)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def research_code_version() -> str:
    return f"research.alpha_evidence:{__version__}"
