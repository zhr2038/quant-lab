from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import polars as pl
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_lab.data.file_index import LAKE_FILE_INDEX, build_lake_file_index
from quant_lab.data.lake import read_parquet_dataset
from quant_lab.factors.factory import (
    COST_BUCKET_DAILY_DATASET,
    FEATURE_VALUE_DATASET,
    MARKET_BAR_DATASET,
)
from quant_lab.factors.plan import build_effective_factor_plan
from quant_lab.research_plane.contracts import (
    FACTOR_FACTORY_SNAPSHOT_SCHEMA,
    FactorFactoryCostSnapshotRecord,
    FactorFactoryPreviousGeneration,
    FactorFactorySnapshotManifest,
    ResearchDatasetReference,
)
from quant_lab.research_plane.signatures import model_content_sha256, sha256_file, sign_model
from quant_lab.research_plane.status import ensure_research_queue_layout

FACTOR_FACTORY_INPUT_DATASETS = (
    FEATURE_VALUE_DATASET,
    MARKET_BAR_DATASET,
    COST_BUCKET_DAILY_DATASET,
)

FEATURE_VALUE_COLUMNS = (
    "feature_set",
    "feature_name",
    "feature_version",
    "symbol",
    "timeframe",
    "ts",
    "value",
    "lookback_bars",
    "input_dataset_version",
    "input_hash",
    "code_version",
    "created_at",
    "source",
    "is_valid",
    "invalid_reason",
)
MARKET_BAR_COLUMNS = (
    "symbol",
    "timeframe",
    "ts",
    "close",
    "is_closed",
)
COST_COLUMNS = (
    "as_of_date",
    "day",
    "symbol",
    "total_cost_bps_p50",
    "total_cost_bps_p75",
    "total_cost_bps_p90",
    "total_cost_bps_p95",
    "cost_model_version",
    "cost_source",
    "source",
)


def seal_factor_factory_snapshot(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    as_of_date: date,
    feature_set: str = "core",
    feature_version: str = "v0.1",
    factor_version: str = "v0.1",
    timeframe: str = "1H",
    horizon_bars: tuple[int, ...] = (4, 8, 24, 72),
    decision_delay_bars: int = 1,
    max_factors: int = 200,
    min_samples: int = 100,
    top_quantile: float = 0.2,
    cost_quantile: str = "p75",
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    quant_lab_commit: str | None = None,
    max_input_bytes: int = 25 * 1024**3,
    max_input_rows: int = 150_000_000,
) -> FactorFactorySnapshotManifest:
    """Build an immutable full-history Factor Factory input snapshot on cloud."""

    root = Path(lake_root).resolve(strict=True)
    queue = ensure_research_queue_layout(queue_root)
    commit = quant_lab_commit or _git_commit()
    horizons = tuple(sorted({int(value) for value in horizon_bars if int(value) > 0}))
    if not horizons:
        raise ValueError("factor_factory_horizon_bars_required")
    if decision_delay_bars < 1:
        raise ValueError("factor_factory_decision_delay_invalid")
    indexed = _load_factor_factory_file_index(root)
    feature_sources = _indexed_dataset_files(root, indexed, FEATURE_VALUE_DATASET)
    market_sources = _indexed_dataset_files(root, indexed, MARKET_BAR_DATASET)
    cost_sources = _indexed_dataset_files(root, indexed, COST_BUCKET_DAILY_DATASET)
    feature_names, feature_min_ts, feature_max_ts = _feature_identity(
        feature_sources,
        feature_set=feature_set,
        feature_version=feature_version,
        timeframe=timeframe,
    )
    plan = build_effective_factor_plan(
        feature_names,
        feature_set=feature_set,
        feature_version=feature_version,
        factor_version=factor_version,
        timeframe=timeframe,
        max_factors=max_factors,
        quant_lab_commit=commit,
        created_at=datetime.combine(as_of_date, time.min, tzinfo=UTC),
    )
    previous_id, previous_digest, previous_manifest = load_factor_factory_generation_binding(root)
    temporary = queue / "snapshots" / f".sealing.{uuid.uuid4().hex}.partial"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        references: list[ResearchDatasetReference] = []
        references.extend(
            _materialize_feature_files(
                root,
                temporary,
                feature_sources,
                feature_set=feature_set,
                feature_version=feature_version,
                timeframe=timeframe,
            )
        )
        market_min_ts: datetime | None = None
        market_max_ts: datetime | None = None
        cost_snapshot: tuple[FactorFactoryCostSnapshotRecord, ...] = ()
        if feature_min_ts is not None:
            if feature_max_ts is None:
                raise RuntimeError("factor_factory_feature_bounds_incomplete")
            market_before = feature_max_ts + _timeframe_delta(timeframe) * (
                decision_delay_bars + max(horizons) + 1
            )
            market_references = _materialize_market_files(
                root,
                temporary,
                market_sources,
                since=feature_min_ts,
                before=market_before,
                timeframe=timeframe,
            )
            references.extend(market_references)
            market_min_ts, market_max_ts = _reference_bounds(market_references)
            cost_references, cost_snapshot = _materialize_cost_selection(
                root,
                temporary,
                cost_sources,
                cost_quantile=cost_quantile,
            )
            references.extend(cost_references)
        references.sort(key=lambda item: (item.dataset_name, item.relative_path))
        total_bytes = sum(item.size_bytes for item in references)
        total_rows = sum(item.row_count for item in references)
        if total_bytes > max_input_bytes:
            raise RuntimeError("factor_factory_snapshot_input_size_limit_exceeded")
        if total_rows > max_input_rows:
            raise RuntimeError("factor_factory_snapshot_input_row_limit_exceeded")
        source_digest = _references_digest(
            references,
            datasets={_dataset_name(FEATURE_VALUE_DATASET), _dataset_name(MARKET_BAR_DATASET)},
        )
        cost_digest = _references_digest(
            references,
            datasets={_dataset_name(COST_BUCKET_DAILY_DATASET)},
        )
        snapshot_seed = model_content_sha256(
            {
                "schema_version": FACTOR_FACTORY_SNAPSHOT_SCHEMA,
                "quant_lab_commit": commit,
                "as_of_date": as_of_date.isoformat(),
                "factor_plan_digest": plan.plan_digest,
                "source_input_digest": source_digest,
                "cost_input_digest": cost_digest,
                "previous_generation_id": previous_id,
                "previous_generation_digest": previous_digest,
                "previous_generation_manifest": (
                    previous_manifest.model_dump(mode="json") if previous_manifest else None
                ),
                "horizon_bars": horizons,
                "decision_delay_bars": decision_delay_bars,
                "min_samples": min_samples,
                "top_quantile": top_quantile,
                "cost_quantile": cost_quantile,
                "result_mode": "PARITY_FULL",
                "history_mode": "bootstrap_full",
            }
        )[:24]
        snapshot_id = f"factor-factory-{snapshot_seed}"
        final_root = queue / "snapshots" / snapshot_id
        if final_root.exists():
            shutil.rmtree(temporary)
            manifest = FactorFactorySnapshotManifest.model_validate_json(
                (final_root / "manifest.json").read_text(encoding="utf-8")
            )
            verify_factor_factory_snapshot_manifest(manifest, final_root=final_root)
            return manifest
        provisional = FactorFactorySnapshotManifest(
            snapshot_id=snapshot_id,
            generated_at=datetime.now(UTC),
            quant_lab_commit=commit,
            as_of_date=as_of_date,
            feature_set=feature_set,
            feature_version=feature_version,
            factor_version=factor_version,
            timeframe=timeframe,
            horizon_bars=horizons,
            decision_delay_bars=decision_delay_bars,
            max_factors=max_factors,
            min_samples=min_samples,
            top_quantile=top_quantile,
            cost_quantile=cost_quantile,
            factor_plan=plan,
            factor_plan_digest=plan.plan_digest,
            source_input_digest=source_digest,
            cost_input_digest=cost_digest,
            cost_snapshot=cost_snapshot,
            feature_min_ts=feature_min_ts,
            feature_max_ts=feature_max_ts,
            market_min_ts=market_min_ts,
            market_max_ts=market_max_ts,
            previous_generation_id=previous_id,
            previous_generation_digest=previous_digest,
            previous_generation_manifest=previous_manifest,
            datasets=[_dataset_name(path) for path in FACTOR_FACTORY_INPUT_DATASETS],
            files=references,
            total_input_bytes=total_bytes,
            total_input_rows=total_rows,
            manifest_sha256="0" * 64,
            signature_key_id=signature_key_id,
            signature="pending",
        )
        digest = model_content_sha256(provisional, blank_fields=("manifest_sha256",))
        unsigned = provisional.model_copy(update={"manifest_sha256": digest})
        manifest = unsigned.model_copy(update={"signature": sign_model(unsigned, signing_key)})
        (temporary / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        (temporary / "SEALED").write_text(digest + "\n", encoding="ascii")
        _make_read_only(temporary)
        os.replace(temporary, final_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_factor_factory_snapshot_manifest(
    manifest: FactorFactorySnapshotManifest,
    *,
    final_root: Path | None = None,
) -> None:
    expected = model_content_sha256(manifest, blank_fields=("manifest_sha256",))
    if expected != manifest.manifest_sha256:
        raise ValueError("factor_factory_snapshot_manifest_digest_mismatch")
    expected_datasets = {_dataset_name(path) for path in FACTOR_FACTORY_INPUT_DATASETS}
    if set(manifest.datasets) != expected_datasets:
        raise ValueError("factor_factory_snapshot_dataset_set_mismatch")
    if final_root is None:
        return
    root = final_root.resolve(strict=True)
    if (root / "SEALED").read_text(encoding="ascii").strip() != manifest.manifest_sha256:
        raise ValueError("factor_factory_snapshot_seal_mismatch")
    for reference in manifest.files:
        unresolved = root / "files" / reference.relative_path
        if _path_has_symlink(root, unresolved):
            raise ValueError("factor_factory_snapshot_path_escape")
        candidate = unresolved.resolve(strict=True)
        if root not in candidate.parents:
            raise ValueError("factor_factory_snapshot_path_escape")
        if candidate.stat().st_size != reference.size_bytes:
            raise ValueError("factor_factory_snapshot_size_mismatch")
        if sha256_file(candidate) != reference.sha256:
            raise ValueError("factor_factory_snapshot_sha256_mismatch")


def load_factor_factory_generation_binding(
    root: Path,
) -> tuple[str | None, str | None, FactorFactoryPreviousGeneration | None]:
    pointer = root / "gold" / "factor_factory_generation.json"
    if not pointer.exists():
        return None, None, None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("factor_factory_generation_pointer_invalid") from exc
    generation_id = str(payload.get("generation_id") or "").strip()
    generation_digest = str(payload.get("generation_digest") or "").strip()
    if not generation_id or len(generation_digest) != 64:
        raise RuntimeError("factor_factory_generation_pointer_incomplete")
    fields = {
        "schema_version",
        "generation_id",
        "generation_digest",
        "task_id",
        "snapshot_id",
        "quant_lab_commit",
        "factor_plan_digest",
        "source_input_digest",
        "cost_input_digest",
        "feature_set",
        "feature_version",
        "factor_version",
        "timeframe",
        "as_of_date",
        "row_counts",
        "dataset_hashes",
        "published_at",
        "diagnostic_only",
        "research_only",
        "live_order_effect",
        "automatic_promotion",
        "max_live_notional_usdt",
    }
    try:
        manifest = FactorFactoryPreviousGeneration.model_validate(
            {field: payload.get(field) for field in fields}
        )
    except ValueError as exc:
        raise RuntimeError("factor_factory_generation_pointer_incomplete") from exc
    return generation_id, generation_digest, manifest


def _load_factor_factory_file_index(root: Path) -> pl.DataFrame:
    names = [_dataset_name(path) for path in FACTOR_FACTORY_INPUT_DATASETS]
    build_lake_file_index(root, names)
    index = read_parquet_dataset(root / LAKE_FILE_INDEX)
    required = {"dataset", "path", "min_ts", "max_ts"}
    if index.is_empty():
        return pl.DataFrame(schema={name: pl.Utf8 for name in sorted(required)})
    if not required.issubset(index.columns):
        raise RuntimeError("lake_file_index_missing_or_invalid")
    return index


def _indexed_dataset_files(root: Path, index: pl.DataFrame, dataset: Path) -> list[Path]:
    name = _dataset_name(dataset)
    files: list[Path] = []
    if index.is_empty():
        return files
    for value in index.filter(pl.col("dataset") == name).get_column("path").to_list():
        relative = str(value or "")
        if not relative:
            continue
        unresolved = root / relative
        if _path_has_symlink(root, unresolved):
            raise ValueError("lake_file_index_path_escape")
        candidate = unresolved.resolve(strict=True)
        dataset_root = (root / dataset).resolve(strict=True)
        if root not in dataset_root.parents or dataset_root not in candidate.parents:
            raise ValueError("lake_file_index_path_escape")
        if candidate.suffix != ".parquet":
            raise ValueError("lake_file_index_non_parquet")
        files.append(candidate)
    return sorted(set(files))


def _feature_identity(
    sources: list[Path],
    *,
    feature_set: str,
    feature_version: str,
    timeframe: str,
) -> tuple[list[str], datetime | None, datetime | None]:
    if not sources:
        return [], None, None
    _require_columns(
        sources,
        {
            "feature_set",
            "feature_name",
            "feature_version",
            "timeframe",
            "symbol",
            "ts",
            "value",
            "is_valid",
        },
        "feature_value",
    )
    lazy = _scan_sources(sources).filter(
        (pl.col("feature_set") == feature_set)
        & (pl.col("feature_version") == feature_version)
        & (pl.col("timeframe") == timeframe)
    )
    lazy = lazy.filter(pl.col("is_valid").cast(pl.Boolean, strict=False).fill_null(False))
    identity = lazy.select(
        _utc_expr("ts").min().alias("min_ts"),
        _utc_expr("ts").max().alias("max_ts"),
    ).collect(engine="streaming")
    names = (
        lazy.select(pl.col("feature_name").cast(pl.Utf8))
        .unique()
        .sort("feature_name")
        .collect(engine="streaming")
        .get_column("feature_name")
        .drop_nulls()
        .to_list()
    )
    return names, _as_utc(identity.item(0, "min_ts")), _as_utc(identity.item(0, "max_ts"))


def _materialize_feature_files(
    root: Path,
    temporary: Path,
    sources: list[Path],
    *,
    feature_set: str,
    feature_version: str,
    timeframe: str,
) -> list[ResearchDatasetReference]:
    references: list[ResearchDatasetReference] = []
    for ordinal, source in enumerate(sources):
        schema = pl.read_parquet_schema(source)
        columns = [column for column in FEATURE_VALUE_COLUMNS if column in schema]
        lazy = pl.scan_parquet(source).filter(
            (pl.col("feature_set") == feature_set)
            & (pl.col("feature_version") == feature_version)
            & (pl.col("timeframe") == timeframe)
        )
        if "is_valid" in schema:
            lazy = lazy.filter(pl.col("is_valid").cast(pl.Boolean, strict=False).fill_null(False))
        reference = _materialize_lazy_source(
            root,
            temporary,
            source,
            dataset=FEATURE_VALUE_DATASET,
            ordinal=ordinal,
            lazy=lazy.select(columns),
        )
        if reference is not None:
            references.append(reference)
    return references


def _materialize_market_files(
    root: Path,
    temporary: Path,
    sources: list[Path],
    *,
    since: datetime,
    before: datetime,
    timeframe: str,
) -> list[ResearchDatasetReference]:
    if not sources:
        return []
    _require_columns(
        sources,
        {"symbol", "timeframe", "ts", "close", "is_closed"},
        "market_bar",
    )
    references: list[ResearchDatasetReference] = []
    for ordinal, source in enumerate(sources):
        schema = pl.read_parquet_schema(source)
        columns = [column for column in MARKET_BAR_COLUMNS if column in schema]
        lazy = pl.scan_parquet(source).filter(
            (pl.col("timeframe") == timeframe)
            & (_utc_expr("ts") >= since)
            & (_utc_expr("ts") < before)
        )
        lazy = lazy.filter(pl.col("is_closed").cast(pl.Boolean, strict=False).fill_null(False))
        reference = _materialize_lazy_source(
            root,
            temporary,
            source,
            dataset=MARKET_BAR_DATASET,
            ordinal=ordinal,
            lazy=lazy.select(columns),
        )
        if reference is not None:
            references.append(reference)
    return references


def _materialize_cost_selection(
    root: Path,
    temporary: Path,
    sources: list[Path],
    *,
    cost_quantile: str,
) -> tuple[list[ResearchDatasetReference], tuple[FactorFactoryCostSnapshotRecord, ...]]:
    if not sources:
        return [], ()
    cost_column = f"total_cost_bps_{cost_quantile}"
    _require_columns(sources, {"symbol", cost_column}, "cost_bucket_daily")
    schema = pl.read_parquet_schema(sources[0])
    if "day" not in schema and "as_of_date" not in schema:
        raise ValueError("factor_factory_cost_bucket_daily_columns_missing:day_or_as_of_date")
    columns = [column for column in COST_COLUMNS if column in schema]
    source_stats = {source: source.stat() for source in sources}
    lazy = _scan_sources(sources)
    day_column = "day" if "day" in schema else "as_of_date" if "as_of_date" in schema else None
    if day_column is not None:
        lazy = lazy.sort(day_column)
    selected = lazy.select(columns).unique(subset=["symbol"], keep="last")
    destination = temporary / "files" / COST_BUCKET_DAILY_DATASET / "part-selected.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.sink_parquet(destination, compression="zstd")
    for source, before in source_stats.items():
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("snapshot_source_changed_while_sealing")
    rows = _parquet_row_count(destination)
    if rows == 0:
        destination.unlink()
        return [], ()
    destination.chmod(0o440)
    stat = destination.stat()
    source_stat = max((item.stat() for item in sources), key=lambda item: item.st_mtime_ns)
    reference = ResearchDatasetReference(
            dataset_name=_dataset_name(COST_BUCKET_DAILY_DATASET),
            source_relative_path=_dataset_name(COST_BUCKET_DAILY_DATASET),
            relative_path=f"{_dataset_name(COST_BUCKET_DAILY_DATASET)}/part-selected.parquet",
            sha256=sha256_file(destination),
            size_bytes=stat.st_size,
            row_count=rows,
            mtime_ns=source_stat.st_mtime_ns,
        )
    selected_frame = pl.read_parquet(destination)
    date_column = "day" if "day" in selected_frame.columns else "as_of_date"
    source_column = "cost_source" if "cost_source" in selected_frame.columns else "source"
    records = tuple(
        FactorFactoryCostSnapshotRecord(
            symbol=str(row["symbol"]),
            cost_date=(str(row[date_column]) if row.get(date_column) is not None else None),
            cost_model_version=str(row.get("cost_model_version") or "unknown"),
            cost_source=str(row.get(source_column) or "unknown"),
            cost_quantile=cost_quantile,
            cost_bps=float(row[cost_column]),
        )
        for row in selected_frame.sort("symbol").to_dicts()
    )
    return [reference], records


def _materialize_lazy_source(
    root: Path,
    temporary: Path,
    source: Path,
    *,
    dataset: Path,
    ordinal: int,
    lazy: pl.LazyFrame,
) -> ResearchDatasetReference | None:
    before = source.stat()
    source_relative = str(source.relative_to(root)).replace("\\", "/")
    part_id = model_content_sha256(
        {"source": source_relative, "ordinal": ordinal, "mtime_ns": before.st_mtime_ns}
    )[:16]
    relative_path = f"{_dataset_name(dataset)}/part-{part_id}.parquet"
    destination = temporary / "files" / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    lazy.sink_parquet(destination, compression="zstd")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("snapshot_source_changed_while_sealing")
    row_count = _parquet_row_count(destination)
    if row_count == 0:
        destination.unlink()
        return None
    destination.chmod(0o440)
    min_ts, max_ts = _parquet_bounds(destination, "ts")
    stat = destination.stat()
    return ResearchDatasetReference(
        dataset_name=_dataset_name(dataset),
        source_relative_path=source_relative,
        relative_path=relative_path,
        sha256=sha256_file(destination),
        size_bytes=stat.st_size,
        row_count=row_count,
        mtime_ns=before.st_mtime_ns,
        min_ts=min_ts,
        max_ts=max_ts,
    )


def _references_digest(
    references: list[ResearchDatasetReference],
    *,
    datasets: set[str],
) -> str:
    return model_content_sha256(
        {
            "schema_version": "quant_lab.factor_factory_input_identity.v1",
            "files": [
                item.model_dump(mode="json") for item in references if item.dataset_name in datasets
            ],
        }
    )


def _reference_bounds(
    references: list[ResearchDatasetReference],
) -> tuple[datetime | None, datetime | None]:
    minima = [item.min_ts for item in references if item.min_ts is not None]
    maxima = [item.max_ts for item in references if item.max_ts is not None]
    return (min(minima) if minima else None, max(maxima) if maxima else None)


def _require_columns(sources: list[Path], required: set[str], dataset: str) -> None:
    for source in sources:
        missing = sorted(required - set(pl.read_parquet_schema(source)))
        if missing:
            raise ValueError(f"factor_factory_{dataset}_columns_missing:{','.join(missing)}")


def _scan_sources(sources: list[Path]) -> pl.LazyFrame:
    return pl.scan_parquet([str(path) for path in sources])


def _parquet_row_count(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item())


def _parquet_bounds(path: Path, column: str) -> tuple[datetime | None, datetime | None]:
    schema = pl.read_parquet_schema(path)
    if column not in schema:
        return None, None
    frame = (
        pl.scan_parquet(path)
        .select(_utc_expr(column).min().alias("minimum"), _utc_expr(column).max().alias("maximum"))
        .collect(engine="streaming")
    )
    return _as_utc(frame.item(0, "minimum")), _as_utc(frame.item(0, "maximum"))


def _utc_expr(column: str) -> pl.Expr:
    return pl.coalesce(
        pl.col(column).cast(pl.Datetime(time_zone="UTC"), strict=False),
        pl.col(column).cast(pl.Utf8, strict=False).str.to_datetime(time_zone="UTC", strict=False),
    )


def _as_utc(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _timeframe_delta(timeframe: str) -> timedelta:
    value = timeframe.strip().lower()
    if len(value) < 2 or not value[:-1].isdigit():
        raise ValueError(f"unsupported factor factory timeframe: {timeframe}")
    amount = int(value[:-1])
    multipliers = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    unit = value[-1]
    if amount <= 0 or unit not in multipliers:
        raise ValueError(f"unsupported factor factory timeframe: {timeframe}")
    return timedelta(seconds=amount * multipliers[unit])


def _dataset_name(path: Path) -> str:
    return str(path).replace("\\", "/")


def _path_has_symlink(root: Path, candidate: Path) -> bool:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _make_read_only(path: Path) -> None:
    for candidate in sorted(path.rglob("*"), reverse=True):
        try:
            candidate.chmod(0o440 if candidate.is_file() else 0o550)
        except OSError:
            pass
    try:
        path.chmod(0o550)
    except OSError:
        pass


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
