from __future__ import annotations

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
from quant_lab.research.alpha_factory.factory import (
    ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET,
    FACTOR_CANDIDATE_DATASET,
    FACTOR_VALUE_DATASET,
    TEMPLATE_REGISTRY_SCHEMA,
    alpha_factory_template_registry_digest,
)
from quant_lab.research.alpha_factory.factory import (
    SCHEMA_VERSION as ALPHA_FACTORY_SCHEMA_VERSION,
)
from quant_lab.research.entry_quality import (
    COST_BUCKET_DAILY_DATASET,
    ENTRY_QUALITY_SCHEMA_VERSION,
    MARKET_BAR_DATASET,
    PULLBACK_HORIZON_HOURS,
    V5_CANDIDATE_EVENT_DATASET,
    V5_CANDIDATE_LABEL_DATASET,
    V5_ORDER_LIFECYCLE_DATASET,
    V5_TRADE_EVENT_DATASET,
)
from quant_lab.research.factor_research.contracts import (
    SCHEMA_VERSION as FACTOR_RESEARCH_SCHEMA_VERSION,
)
from quant_lab.research.factor_research.contracts import ResearchHypothesis
from quant_lab.research.factor_research.registry import (
    HYPOTHESIS_REGISTRY_SCHEMA,
    RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
    RESEARCH_TRIAL_LEDGER_DATASET,
    TRIAL_LEDGER_SCHEMA,
    hypothesis_registry_digest,
    trial_ledger_digest,
)
from quant_lab.research.second_stage_alpha_factory import (
    BTC_EXIT_REVIEW_DATASET,
    DEFAULT_HORIZONS,
    EXPANDED_QUALITY_DATASET,
    MARKET_REGIME_DATASET,
    PAPER_RUN_DATASET,
)
from quant_lab.research.second_stage_alpha_factory import (
    SCHEMA_VERSION as SECOND_STAGE_SCHEMA_VERSION,
)
from quant_lab.research.strategy_evidence import (
    SAMPLE_SCHEMA as STRATEGY_EVIDENCE_SAMPLE_SCHEMA,
)
from quant_lab.research.strategy_evidence import (
    SUMMARY_SCHEMA as STRATEGY_EVIDENCE_SUMMARY_SCHEMA,
)
from quant_lab.research_plane.contracts import (
    ALPHA_FACTORY_SNAPSHOT_SCHEMA,
    FACTOR_RESEARCH_SNAPSHOT_SCHEMA,
    RESEARCH_SNAPSHOT_SCHEMA,
    AlphaFactorySnapshotManifest,
    FactorResearchSnapshotManifest,
    ResearchDatasetReference,
    ResearchSnapshotManifest,
)
from quant_lab.research_plane.signatures import model_content_sha256, sha256_file, sign_model
from quant_lab.research_plane.status import ensure_research_queue_layout
from quant_lab.symbols import normalize_symbol

ENTRY_QUALITY_INPUT_DATASETS = (
    V5_TRADE_EVENT_DATASET,
    V5_ORDER_LIFECYCLE_DATASET,
    MARKET_BAR_DATASET,
    V5_CANDIDATE_EVENT_DATASET,
    V5_CANDIDATE_LABEL_DATASET,
    COST_BUCKET_DAILY_DATASET,
)

STRATEGY_EVIDENCE_DATASET = Path("gold") / "strategy_evidence"
STRATEGY_EVIDENCE_SAMPLE_DATASET = Path("gold") / "strategy_evidence_sample"

ALPHA_FACTORY_INPUT_DATASETS = (
    MARKET_BAR_DATASET,
    EXPANDED_QUALITY_DATASET,
    COST_BUCKET_DAILY_DATASET,
    MARKET_REGIME_DATASET,
    BTC_EXIT_REVIEW_DATASET,
    PAPER_RUN_DATASET,
    STRATEGY_EVIDENCE_DATASET,
    STRATEGY_EVIDENCE_SAMPLE_DATASET,
    FACTOR_CANDIDATE_DATASET,
    FACTOR_VALUE_DATASET,
    ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET,
)

ALPHA_FACTORY_COLUMN_PROJECTIONS: dict[str, tuple[str, ...]] = {
    "silver/market_bar": (
        "symbol",
        "timeframe",
        "ts",
        "ts_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "is_closed",
        "ingest_ts",
    ),
    "gold/expanded_universe_quality": (
        "as_of_date",
        "created_at",
        "symbol",
        "symbol_quality_score",
        "quality_score",
        "avg_spread_bps",
        "spread_bps_p75",
        "spread_bps",
        "avg_spread",
        "btc_correlation",
        "btc_corr",
        "correlation_to_btc",
        "volume_24h_usdt",
        "quote_volume_24h",
        "quote_volume_24h_usdt",
        "volume_usdt_24h",
    ),
    "gold/cost_bucket_daily": (
        "as_of_date",
        "day",
        "created_at",
        "symbol",
        "roundtrip_all_in_cost_bps",
        "selected_total_cost_bps",
        "total_cost_bps_p75",
        "cost_bps",
        "selected_entry_gate_cost_bps",
        "fee_bps_p75",
        "slippage_bps_p75",
        "spread_bps_p75",
        "source",
        "cost_source",
        "fallback_level",
    ),
    "gold/market_regime_daily": (
        "as_of_date",
        "as_of_ts",
        "created_at",
        "ts",
        "date",
        "symbol",
        "current_regime",
        "regime_state",
        "market_regime",
        "state",
    ),
    "gold/btc_probe_exit_policy_review": (
        "as_of_date",
        "created_at",
        "entry_ts",
        "ts_utc",
        "run_id",
        "trade_id",
        "entry_px",
        "entry_price",
        "label_close",
        "exit_px",
        "actual_exit_net_bps",
        "realized_net_bps",
        "exit_net_bps",
        "net_bps",
        "fixed_hold_4h_net_bps",
        "fixed_hold_8h_net_bps",
        "fixed_hold_12h_net_bps",
        "fixed_hold_24h_net_bps",
        "fixed_hold_48h_net_bps",
        "would_hold_4h_net_bps",
        "would_hold_8h_net_bps",
        "would_hold_12h_net_bps",
        "would_hold_24h_net_bps",
        "would_hold_48h_net_bps",
        "would_have_held_4h_net_bps",
        "would_have_held_8h_net_bps",
        "would_have_held_12h_net_bps",
        "would_have_held_24h_net_bps",
        "would_have_held_48h_net_bps",
        "mfe_bps",
        "mae_bps",
        "exit_reason",
    ),
    "gold/paper_strategy_runs": (
        "as_of_date",
        "created_at",
        "ts_utc",
        "strategy_id",
        "proposal_id",
        "run_id",
        "source_entry_id",
        "actual_exit_net_bps",
        "paper_pnl_bps",
        "paper_pnl_bps_4h",
        "paper_pnl_bps_8h",
        "paper_pnl_bps_12h",
        "paper_pnl_bps_24h",
        "paper_pnl_bps_48h",
        "realized_net_bps",
        "net_bps",
        "fixed_hold_4h_net_bps",
        "fixed_hold_8h_net_bps",
        "fixed_hold_12h_net_bps",
        "fixed_hold_24h_net_bps",
        "fixed_hold_48h_net_bps",
        "mfe_bps",
        "mfe_bps_4h",
        "mfe_bps_8h",
        "mfe_bps_12h",
        "mfe_bps_24h",
        "mfe_bps_48h",
        "mae_bps",
        "mae_bps_4h",
        "mae_bps_8h",
        "mae_bps_12h",
        "mae_bps_24h",
        "mae_bps_48h",
        "cost_source",
        "regime_state",
        "exit_reason",
    ),
    "gold/strategy_evidence": tuple(STRATEGY_EVIDENCE_SUMMARY_SCHEMA),
    "gold/strategy_evidence_sample": tuple(STRATEGY_EVIDENCE_SAMPLE_SCHEMA),
    "gold/factor_candidate": (
        "as_of_date",
        "hypothesis_id",
        "hypothesis_version",
        "data_snapshot_id",
        "factor_id",
        "factor_family",
        "correlation_cluster_id",
        "candidate_state",
        "signal_validity",
        "portfolio_validity",
        "deployment_readiness",
        "research_only",
        "live_order_effect",
        "automatic_promotion",
        "max_live_notional_usdt",
    ),
    "gold/factor_value": (
        "hypothesis_id",
        "hypothesis_version",
        "feature_recipe_id",
        "data_snapshot_id",
        "factor_id",
        "symbol",
        "timeframe",
        "ts",
        "feature_ts",
        "available_time",
        "decision_ts",
        "created_at",
        "value",
        "normalized_value",
        "rank_value",
        "raw_value",
        "is_valid",
        "regime",
        "regime_state",
        "market_regime",
        "current_regime",
        "research_only",
        "live_order_effect",
    ),
    "gold/alpha_factory_template_registry": tuple(TEMPLATE_REGISTRY_SCHEMA),
}

FACTOR_RESEARCH_DATA_DATASETS = (
    MARKET_BAR_DATASET,
    COST_BUCKET_DAILY_DATASET,
    MARKET_REGIME_DATASET,
)

FACTOR_RESEARCH_INPUT_DATASETS = (
    *FACTOR_RESEARCH_DATA_DATASETS,
    RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
    RESEARCH_TRIAL_LEDGER_DATASET,
)

FACTOR_RESEARCH_COLUMN_PROJECTIONS: dict[str, tuple[str, ...]] = {
    "silver/market_bar": (
        "symbol",
        "timeframe",
        "ts",
        "ts_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "is_closed",
        "ingest_ts",
    ),
    "gold/cost_bucket_daily": (
        "as_of_date",
        "day",
        "created_at",
        "symbol",
        "roundtrip_all_in_cost_bps",
        "selected_total_cost_bps",
        "total_cost_bps_p75",
        "cost_bps",
        "cost_source",
        "source",
        "fallback_level",
    ),
    "gold/market_regime_daily": (
        "as_of_date",
        "day",
        "as_of_ts",
        "created_at",
        "ts",
        "date",
        "symbol",
        "current_regime",
        "regime_state",
        "market_regime",
        "state",
    ),
}

SNAPSHOT_TIME_COLUMNS = {
    str(V5_TRADE_EVENT_DATASET).replace("\\", "/"): ("ts_utc", "ts", "entry_ts"),
    str(V5_ORDER_LIFECYCLE_DATASET).replace("\\", "/"): (
        "ts_utc",
        "last_fill_ts",
        "submit_ts",
        "decision_ts",
    ),
    str(MARKET_BAR_DATASET).replace("\\", "/"): ("ts", "ts_utc"),
    str(V5_CANDIDATE_EVENT_DATASET).replace("\\", "/"): ("ts_utc", "ts"),
    str(V5_CANDIDATE_LABEL_DATASET).replace("\\", "/"): (
        "decision_ts",
        "ts_utc",
    ),
    str(COST_BUCKET_DAILY_DATASET).replace("\\", "/"): ("as_of_date", "day", "created_at"),
    str(EXPANDED_QUALITY_DATASET).replace("\\", "/"): ("as_of_date", "created_at"),
    str(MARKET_REGIME_DATASET).replace("\\", "/"): (
        "as_of_ts",
        "created_at",
        "ts",
        "as_of_date",
        "day",
        "date",
    ),
    str(BTC_EXIT_REVIEW_DATASET).replace("\\", "/"): (
        "entry_ts",
        "created_at",
        "as_of_date",
    ),
    str(PAPER_RUN_DATASET).replace("\\", "/"): (
        "ts_utc",
        "created_at",
        "as_of_date",
    ),
    str(STRATEGY_EVIDENCE_DATASET).replace("\\", "/"): (
        "as_of_date",
        "created_at",
    ),
    str(STRATEGY_EVIDENCE_SAMPLE_DATASET).replace("\\", "/"): (
        "sample_ts",
        "created_at",
        "as_of_date",
    ),
    str(FACTOR_CANDIDATE_DATASET).replace("\\", "/"): ("as_of_date",),
    str(FACTOR_VALUE_DATASET).replace("\\", "/"): (
        "available_time",
        "decision_ts",
        "ts",
        "feature_ts",
        "created_at",
    ),
}


def seal_entry_quality_history_snapshot(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    start_date: date,
    end_date: date,
    selected_v5_bundle_id: str,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    quant_lab_commit: str | None = None,
) -> ResearchSnapshotManifest:
    root = Path(lake_root).resolve()
    queue = ensure_research_queue_layout(queue_root)
    commit = quant_lab_commit or _git_commit()
    start_dt = datetime.combine(start_date, time.min, tzinfo=UTC)
    end_dt = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=UTC)
    windows = {
        str(V5_TRADE_EVENT_DATASET).replace("\\", "/"): (start_dt, end_dt),
        str(V5_ORDER_LIFECYCLE_DATASET).replace("\\", "/"): (start_dt, end_dt),
        str(V5_CANDIDATE_EVENT_DATASET).replace("\\", "/"): (start_dt, end_dt),
        str(MARKET_BAR_DATASET).replace("\\", "/"): (
            start_dt - timedelta(hours=24),
            end_dt + timedelta(hours=max(PULLBACK_HORIZON_HOURS)),
        ),
        str(V5_CANDIDATE_LABEL_DATASET).replace("\\", "/"): (
            start_dt,
            end_dt + timedelta(hours=max(PULLBACK_HORIZON_HOURS)),
        ),
        str(COST_BUCKET_DAILY_DATASET).replace("\\", "/"): (start_dt, end_dt),
    }
    index = _load_required_file_index(root)
    selected = _select_indexed_files(root, index, windows)
    temporary = queue / "snapshots" / f".sealing.{uuid.uuid4().hex}.partial"
    temporary.mkdir(parents=True, exist_ok=False)
    references: list[ResearchDatasetReference] = []
    try:
        for dataset, indexed_source, min_ts, max_ts in selected:
            if indexed_source.is_symlink():
                raise ValueError("snapshot_source_path_escape")
            source = indexed_source.resolve(strict=True)
            if root not in source.parents:
                raise ValueError("snapshot_source_path_escape")
            source_relative = str(source.relative_to(root)).replace("\\", "/")
            relative_path = source_relative
            destination = temporary / "files" / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            before = source.stat()
            shutil.copy2(source, destination)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError("snapshot_source_changed_while_sealing")
            destination.chmod(0o440)
            stat = destination.stat()
            references.append(
                ResearchDatasetReference(
                    dataset_name=dataset,
                    source_relative_path=source_relative,
                    relative_path=relative_path,
                    sha256=sha256_file(destination),
                    size_bytes=stat.st_size,
                    row_count=_parquet_row_count(destination),
                    mtime_ns=stat.st_mtime_ns,
                    min_ts=min_ts,
                    max_ts=max_ts,
                )
            )
        references.sort(key=lambda item: (item.dataset_name, item.relative_path))
        snapshot_seed = model_content_sha256(
            {
                "schema_version": RESEARCH_SNAPSHOT_SCHEMA,
                "commit": commit,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "selected_v5_bundle_id": selected_v5_bundle_id,
                "entry_quality_schema_version": ENTRY_QUALITY_SCHEMA_VERSION,
                "files": [item.model_dump(mode="json") for item in references],
            }
        )[:24]
        snapshot_id = f"entry-quality-history-{snapshot_seed}"
        final_root = queue / "snapshots" / snapshot_id
        if (final_root / "SEALED").is_file():
            manifest = ResearchSnapshotManifest.model_validate_json(
                (final_root / "manifest.json").read_text(encoding="utf-8")
            )
            released = final_root / "FILES_RELEASED.json"
            if released.is_file():
                if manifest.files != references:
                    raise RuntimeError("research_snapshot_rehydrate_source_mismatch")
                _make_snapshot_writable(final_root)
                existing_payload_valid = False
                if (final_root / "files").is_dir():
                    try:
                        verify_snapshot_manifest(manifest, final_root=final_root)
                        existing_payload_valid = True
                    except (OSError, ValueError):
                        shutil.rmtree(final_root / "files", ignore_errors=True)
                source_files = temporary / "files"
                if references and not source_files.is_dir() and not existing_payload_valid:
                    raise RuntimeError("research_snapshot_rehydrate_payload_missing")
                if source_files.exists() and not existing_payload_valid:
                    os.replace(source_files, final_root / "files")
                released.chmod(0o640)
                released.unlink()
                _make_snapshot_read_only(final_root)
            verify_snapshot_manifest(manifest, final_root=final_root)
            shutil.rmtree(temporary, ignore_errors=True)
            return manifest
        if final_root.exists():
            raise RuntimeError("research_snapshot_destination_incomplete")
        provisional = ResearchSnapshotManifest(
            schema_version=RESEARCH_SNAPSHOT_SCHEMA,
            snapshot_id=snapshot_id,
            generated_at=datetime.now(UTC),
            quant_lab_commit=commit,
            selected_v5_bundle_id=selected_v5_bundle_id,
            entry_quality_schema_version=ENTRY_QUALITY_SCHEMA_VERSION,
            datasets=[str(path).replace("\\", "/") for path in ENTRY_QUALITY_INPUT_DATASETS],
            files=references,
            total_input_bytes=sum(item.size_bytes for item in references),
            total_input_rows=sum(item.row_count for item in references),
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
        _make_snapshot_read_only(temporary)
        os.replace(temporary, final_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def seal_alpha_factory_snapshot(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    as_of_date: date,
    lookback_days: int,
    max_candidates: int,
    selected_v5_bundle_id: str,
    effective_registry: pl.DataFrame,
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    quant_lab_commit: str | None = None,
    factor_generation_binding: dict[str, object] | None = None,
    max_input_bytes: int = 25 * 1024**3,
    max_input_rows: int = 50_000_000,
) -> AlphaFactorySnapshotManifest:
    """Seal projected Alpha Factory inputs without scanning or copying unbounded history."""
    if not 1 <= lookback_days <= 180:
        raise ValueError("alpha_factory_lookback_days_out_of_range")
    if not 1 <= max_candidates <= 200:
        raise ValueError("alpha_factory_max_candidates_out_of_range")
    root = Path(lake_root).resolve()
    queue = ensure_research_queue_layout(queue_root)
    commit = quant_lab_commit or _git_commit()
    binding = dict(factor_generation_binding or {})
    expected_binding_fields = {
        "factor_generation_id",
        "factor_generation_digest",
        "factor_generation_as_of_date",
        "factor_generation_published_at",
        "hypothesis_registry_digest",
        "trial_ledger_digest",
        "factor_generation_fresh",
        "factor_generation_hypothesis_ids",
    }
    if binding and set(binding) != expected_binding_fields:
        raise ValueError("alpha_factory_factor_generation_binding_incomplete")
    registry_digest = alpha_factory_template_registry_digest(effective_registry)
    day_start = datetime.combine(as_of_date, time.min, tzinfo=UTC)
    label_end = day_start + timedelta(days=1)
    lookback_start = day_start - timedelta(days=lookback_days)
    max_horizon = max(DEFAULT_HORIZONS)
    market_start = lookback_start - timedelta(hours=max_horizon)
    market_end = label_end + timedelta(hours=max_horizon)
    windows = {
        "silver/market_bar": (market_start, market_end),
        "gold/expanded_universe_quality": (lookback_start, label_end),
        "gold/cost_bucket_daily": (lookback_start, label_end),
        "gold/market_regime_daily": (lookback_start, label_end),
        "gold/btc_probe_exit_policy_review": (lookback_start, label_end),
        "gold/paper_strategy_runs": (lookback_start, label_end),
        "gold/strategy_evidence": (day_start, label_end),
        "gold/strategy_evidence_sample": (lookback_start, label_end),
        "gold/factor_candidate": (lookback_start, label_end),
        "gold/factor_value": (lookback_start, label_end),
    }
    indexed_inputs = tuple(
        dataset
        for dataset in ALPHA_FACTORY_INPUT_DATASETS
        if dataset != ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET
    )
    index = _load_required_file_index(root, indexed_inputs)
    selected = _select_indexed_files(root, index, windows)
    quality_dataset = str(EXPANDED_QUALITY_DATASET).replace("\\", "/")
    latest_quality_day = _latest_alpha_dataset_day(
        selected,
        dataset=quality_dataset,
        window=windows[quality_dataset],
    )
    if latest_quality_day is not None:
        selected = _restrict_alpha_dataset_to_day(
            selected,
            dataset=quality_dataset,
            effective_day=latest_quality_day,
        )
    source_identities = _alpha_source_identities(root, selected)
    source_input_digest = model_content_sha256(
        {
            "schema_version": "quant_lab.alpha_factory_source_identity.v1",
            "files": source_identities,
        }
    )
    symbols = _alpha_required_symbols(selected)
    snapshot_seed = model_content_sha256(
        {
            "schema_version": ALPHA_FACTORY_SNAPSHOT_SCHEMA,
            "commit": commit,
            "as_of_date": as_of_date.isoformat(),
            "lookback_days": lookback_days,
            "max_candidates": max_candidates,
            "selected_v5_bundle_id": selected_v5_bundle_id,
            "alpha_factory_schema_version": ALPHA_FACTORY_SCHEMA_VERSION,
            "second_stage_schema_version": SECOND_STAGE_SCHEMA_VERSION,
            "template_registry_digest": registry_digest,
            "factor_generation_binding": {
                key: value.isoformat() if isinstance(value, (date, datetime)) else value
                for key, value in sorted(binding.items())
            },
            "source_input_digest": source_input_digest,
            "symbols": symbols,
            "expanded_quality_effective_day": (
                latest_quality_day.isoformat() if latest_quality_day is not None else None
            ),
            "column_projections": ALPHA_FACTORY_COLUMN_PROJECTIONS,
        }
    )[:24]
    snapshot_id = f"alpha-factory-{snapshot_seed}"
    final_root = queue / "snapshots" / snapshot_id
    if (final_root / "SEALED").is_file() and not (final_root / "FILES_RELEASED.json").is_file():
        manifest = AlphaFactorySnapshotManifest.model_validate_json(
            (final_root / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.source_input_digest != source_input_digest:
            raise RuntimeError("alpha_factory_snapshot_source_identity_mismatch")
        verify_alpha_factory_snapshot_manifest(manifest, final_root=final_root)
        return manifest

    temporary = queue / "snapshots" / f".sealing.{uuid.uuid4().hex}.partial"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        references = _materialize_alpha_factory_snapshot_files(
            root,
            temporary,
            selected=selected,
            windows=windows,
            symbols=symbols,
            effective_registry=effective_registry,
            exact_dataset_days={quality_dataset: latest_quality_day}
            if latest_quality_day is not None
            else {},
        )
        total_bytes = sum(item.size_bytes for item in references)
        total_rows = sum(item.row_count for item in references)
        if total_bytes > max_input_bytes:
            raise RuntimeError("alpha_factory_snapshot_input_size_limit_exceeded")
        if total_rows > max_input_rows:
            raise RuntimeError("alpha_factory_snapshot_input_row_limit_exceeded")
        if (final_root / "SEALED").is_file():
            manifest = AlphaFactorySnapshotManifest.model_validate_json(
                (final_root / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest.source_input_digest != source_input_digest or manifest.files != references:
                raise RuntimeError("alpha_factory_snapshot_rehydrate_source_mismatch")
            _make_snapshot_writable(final_root)
            shutil.rmtree(final_root / "files", ignore_errors=True)
            os.replace(temporary / "files", final_root / "files")
            (final_root / "FILES_RELEASED.json").unlink(missing_ok=True)
            _make_snapshot_read_only(final_root)
            verify_alpha_factory_snapshot_manifest(manifest, final_root=final_root)
            shutil.rmtree(temporary, ignore_errors=True)
            return manifest
        if final_root.exists():
            raise RuntimeError("research_snapshot_destination_incomplete")
        provisional = AlphaFactorySnapshotManifest(
            snapshot_id=snapshot_id,
            generated_at=datetime.now(UTC),
            quant_lab_commit=commit,
            selected_v5_bundle_id=selected_v5_bundle_id,
            alpha_factory_schema_version=ALPHA_FACTORY_SCHEMA_VERSION,
            second_stage_schema_version=SECOND_STAGE_SCHEMA_VERSION,
            template_registry_digest=registry_digest,
            **binding,
            source_input_digest=source_input_digest,
            as_of_date=as_of_date,
            lookback_days=lookback_days,
            max_candidates=max_candidates,
            datasets=[str(path).replace("\\", "/") for path in ALPHA_FACTORY_INPUT_DATASETS],
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
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (temporary / "SEALED").write_text(digest + "\n", encoding="ascii")
        _make_snapshot_read_only(temporary)
        os.replace(temporary, final_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def factor_research_source_identity(
    lake_root: str | Path,
    *,
    start_date: date,
    end_date: date,
    hypotheses: list[ResearchHypothesis],
) -> tuple[str, str]:
    """Return the immutable source digest and provisional data-snapshot identity."""
    root = Path(lake_root).resolve()
    selected, windows, _symbols = _select_factor_research_inputs(
        root,
        start_date=start_date,
        end_date=end_date,
        hypotheses=hypotheses,
    )
    digest = _factor_research_source_digest(root, selected=selected, windows=windows)
    return digest, f"factor-input-{digest[:24]}"


def seal_factor_research_snapshot(
    lake_root: str | Path,
    queue_root: str | Path,
    *,
    as_of_date: date,
    start_date: date,
    end_date: date,
    max_history_days: int,
    selected_v5_bundle_id: str,
    effective_registry: pl.DataFrame,
    trial_ledger: pl.DataFrame,
    hypotheses: list[ResearchHypothesis],
    signing_key: Ed25519PrivateKey,
    signature_key_id: str,
    quant_lab_commit: str | None = None,
    expected_source_input_digest: str | None = None,
    max_input_bytes: int = 25 * 1024**3,
    max_input_rows: int = 100_000_000,
) -> FactorResearchSnapshotManifest:
    """Seal bounded point-in-time inputs and pre-registered trials for NAS research."""
    if end_date < start_date or (end_date - start_date).days > max_history_days:
        raise ValueError("factor_research_window_out_of_range")
    if not hypotheses:
        raise ValueError("factor_research_hypotheses_required")
    root = Path(lake_root).resolve()
    queue = ensure_research_queue_layout(queue_root)
    commit = quant_lab_commit or _git_commit()
    registry_digest = hypothesis_registry_digest(effective_registry)
    ledger_digest = trial_ledger_digest(trial_ledger)
    selected, windows, symbols = _select_factor_research_inputs(
        root,
        start_date=start_date,
        end_date=end_date,
        hypotheses=hypotheses,
    )
    source_input_digest = _factor_research_source_digest(root, selected=selected, windows=windows)
    if expected_source_input_digest and source_input_digest != expected_source_input_digest:
        raise RuntimeError("factor_research_source_identity_changed")
    hypothesis_ids = tuple(sorted(item.hypothesis_id for item in hypotheses))
    trial_ids = tuple(sorted(str(value) for value in trial_ledger.get_column("trial_id").to_list()))
    snapshot_seed = model_content_sha256(
        {
            "schema_version": FACTOR_RESEARCH_SNAPSHOT_SCHEMA,
            "commit": commit,
            "as_of_date": as_of_date.isoformat(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "max_history_days": max_history_days,
            "selected_v5_bundle_id": selected_v5_bundle_id,
            "factor_research_schema_version": FACTOR_RESEARCH_SCHEMA_VERSION,
            "hypothesis_registry_digest": registry_digest,
            "trial_ledger_digest": ledger_digest,
            "source_input_digest": source_input_digest,
            "hypothesis_ids": hypothesis_ids,
            "trial_ids": trial_ids,
            "symbols": symbols,
            "column_projections": FACTOR_RESEARCH_COLUMN_PROJECTIONS,
        }
    )[:24]
    snapshot_id = f"factor-research-{snapshot_seed}"
    final_root = queue / "snapshots" / snapshot_id
    if (final_root / "SEALED").is_file() and not (final_root / "FILES_RELEASED.json").is_file():
        manifest = FactorResearchSnapshotManifest.model_validate_json(
            (final_root / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            manifest.source_input_digest != source_input_digest
            or manifest.hypothesis_registry_digest != registry_digest
            or manifest.trial_ledger_digest != ledger_digest
        ):
            raise RuntimeError("factor_research_snapshot_identity_mismatch")
        verify_factor_research_snapshot_manifest(manifest, final_root=final_root)
        return manifest

    temporary = queue / "snapshots" / f".sealing.{uuid.uuid4().hex}.partial"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        references = _materialize_factor_research_snapshot_files(
            root,
            temporary,
            selected=selected,
            windows=windows,
            symbols=symbols,
            effective_registry=effective_registry,
            trial_ledger=trial_ledger,
        )
        total_bytes = sum(item.size_bytes for item in references)
        total_rows = sum(item.row_count for item in references)
        if total_bytes > max_input_bytes:
            raise RuntimeError("factor_research_snapshot_input_size_limit_exceeded")
        if total_rows > max_input_rows:
            raise RuntimeError("factor_research_snapshot_input_row_limit_exceeded")
        if final_root.exists():
            raise RuntimeError("research_snapshot_destination_incomplete")
        provisional = FactorResearchSnapshotManifest(
            snapshot_id=snapshot_id,
            generated_at=datetime.now(UTC),
            quant_lab_commit=commit,
            selected_v5_bundle_id=selected_v5_bundle_id,
            factor_research_schema_version=FACTOR_RESEARCH_SCHEMA_VERSION,
            hypothesis_registry_digest=registry_digest,
            trial_ledger_digest=ledger_digest,
            source_input_digest=source_input_digest,
            as_of_date=as_of_date,
            start_date=start_date,
            end_date=end_date,
            max_history_days=max_history_days,
            hypothesis_ids=hypothesis_ids,
            trial_ids=trial_ids,
            test_count=len(trial_ids),
            datasets=[str(path).replace("\\", "/") for path in FACTOR_RESEARCH_INPUT_DATASETS],
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
        _make_snapshot_read_only(temporary)
        os.replace(temporary, final_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_snapshot_manifest(
    manifest: ResearchSnapshotManifest,
    *,
    final_root: Path | None = None,
) -> None:
    expected = model_content_sha256(manifest, blank_fields=("manifest_sha256",))
    if expected != manifest.manifest_sha256:
        raise ValueError("research_snapshot_manifest_digest_mismatch")
    expected_datasets = {str(path).replace("\\", "/") for path in ENTRY_QUALITY_INPUT_DATASETS}
    if set(manifest.datasets) != expected_datasets:
        raise ValueError("research_snapshot_dataset_set_mismatch")
    if final_root is not None:
        root = final_root.resolve(strict=True)
        seal = (root / "SEALED").read_text(encoding="ascii").strip()
        if seal != manifest.manifest_sha256:
            raise ValueError("research_snapshot_seal_mismatch")
        for reference in manifest.files:
            unresolved = root / "files" / reference.relative_path
            if _path_has_symlink(root, unresolved):
                raise ValueError("research_snapshot_path_escape")
            candidate = unresolved.resolve(strict=True)
            if root not in candidate.parents:
                raise ValueError("research_snapshot_path_escape")
            if candidate.stat().st_size != reference.size_bytes:
                raise ValueError("research_snapshot_size_mismatch")
            if sha256_file(candidate) != reference.sha256:
                raise ValueError("research_snapshot_sha256_mismatch")


def verify_alpha_factory_snapshot_manifest(
    manifest: AlphaFactorySnapshotManifest,
    *,
    final_root: Path | None = None,
) -> None:
    expected = model_content_sha256(manifest, blank_fields=("manifest_sha256",))
    if expected != manifest.manifest_sha256:
        raise ValueError("alpha_factory_snapshot_manifest_digest_mismatch")
    expected_datasets = {str(path).replace("\\", "/") for path in ALPHA_FACTORY_INPUT_DATASETS}
    if set(manifest.datasets) != expected_datasets:
        raise ValueError("alpha_factory_snapshot_dataset_set_mismatch")
    if final_root is None:
        return
    root = final_root.resolve(strict=True)
    seal = (root / "SEALED").read_text(encoding="ascii").strip()
    if seal != manifest.manifest_sha256:
        raise ValueError("alpha_factory_snapshot_seal_mismatch")
    for reference in manifest.files:
        unresolved = root / "files" / reference.relative_path
        if _path_has_symlink(root, unresolved):
            raise ValueError("alpha_factory_snapshot_path_escape")
        candidate = unresolved.resolve(strict=True)
        if root not in candidate.parents:
            raise ValueError("alpha_factory_snapshot_path_escape")
        if candidate.stat().st_size != reference.size_bytes:
            raise ValueError("alpha_factory_snapshot_size_mismatch")
        if sha256_file(candidate) != reference.sha256:
            raise ValueError("alpha_factory_snapshot_sha256_mismatch")


def verify_factor_research_snapshot_manifest(
    manifest: FactorResearchSnapshotManifest,
    *,
    final_root: Path | None = None,
) -> None:
    expected = model_content_sha256(manifest, blank_fields=("manifest_sha256",))
    if expected != manifest.manifest_sha256:
        raise ValueError("factor_research_snapshot_manifest_digest_mismatch")
    expected_datasets = {str(path).replace("\\", "/") for path in FACTOR_RESEARCH_INPUT_DATASETS}
    if set(manifest.datasets) != expected_datasets:
        raise ValueError("factor_research_snapshot_dataset_set_mismatch")
    if final_root is None:
        return
    root = final_root.resolve(strict=True)
    seal = (root / "SEALED").read_text(encoding="ascii").strip()
    if seal != manifest.manifest_sha256:
        raise ValueError("factor_research_snapshot_seal_mismatch")
    for reference in manifest.files:
        unresolved = root / "files" / reference.relative_path
        if _path_has_symlink(root, unresolved):
            raise ValueError("factor_research_snapshot_path_escape")
        candidate = unresolved.resolve(strict=True)
        if root not in candidate.parents:
            raise ValueError("factor_research_snapshot_path_escape")
        if candidate.stat().st_size != reference.size_bytes:
            raise ValueError("factor_research_snapshot_size_mismatch")
        if sha256_file(candidate) != reference.sha256:
            raise ValueError("factor_research_snapshot_sha256_mismatch")


def _select_factor_research_inputs(
    root: Path,
    *,
    start_date: date,
    end_date: date,
    hypotheses: list[ResearchHypothesis],
) -> tuple[
    list[tuple[str, Path, datetime | None, datetime | None]],
    dict[str, tuple[datetime, datetime]],
    list[str],
]:
    if end_date < start_date:
        raise ValueError("factor_research_end_before_start")
    executable_requirements: dict[str, set[str]] = {}
    max_lookback = 1
    max_horizon = 1
    for hypothesis in hypotheses:
        if not hypothesis.available_data_confirmed or hypothesis.blocked_missing_data:
            raise ValueError(f"factor_research_hypothesis_data_blocked:{hypothesis.hypothesis_id}")
        max_horizon = max(max_horizon, *hypothesis.expected_horizons)
        for recipe in hypothesis.feature_recipes:
            numeric = [int(item.value) for item in recipe.parameters if item.value.isdigit()]
            if numeric:
                max_lookback = max(max_lookback, *numeric)
        for requirement in hypothesis.data_requirements:
            if requirement.dataset_name not in FACTOR_RESEARCH_COLUMN_PROJECTIONS:
                raise ValueError(f"factor_research_unsupported_dataset:{requirement.dataset_name}")
            executable_requirements.setdefault(requirement.dataset_name, set()).update(
                requirement.required_columns
            )
    start = datetime.combine(start_date, time.min, tzinfo=UTC)
    end = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=UTC)
    windows = {
        "silver/market_bar": (
            start - timedelta(hours=max_lookback),
            end + timedelta(hours=max_horizon),
        ),
        "gold/cost_bucket_daily": (start, end),
        "gold/market_regime_daily": (start, end),
    }
    required_paths = tuple(Path(name) for name in sorted(executable_requirements))
    index = _load_required_file_index(root, required_paths)
    selected = [
        item
        for item in _select_indexed_files(root, index, windows)
        if item[0] in executable_requirements
    ]
    selected_by_dataset: dict[str, list[Path]] = {}
    for dataset, source, _min_ts, _max_ts in selected:
        selected_by_dataset.setdefault(dataset, []).append(source)
    for dataset, required_columns in executable_requirements.items():
        sources = selected_by_dataset.get(dataset, [])
        if not sources:
            raise ValueError(f"factor_research_required_dataset_empty:{dataset}")
        for source in sources:
            schema = pl.read_parquet_schema(source)
            missing = sorted(required_columns - set(schema))
            if missing:
                raise ValueError(
                    f"factor_research_required_columns_missing:{dataset}:{','.join(missing)}"
                )
    symbols = _alpha_required_symbols(selected)
    return selected, windows, symbols


def _factor_research_source_digest(
    root: Path,
    *,
    selected: list[tuple[str, Path, datetime | None, datetime | None]],
    windows: dict[str, tuple[datetime, datetime]],
) -> str:
    return model_content_sha256(
        {
            "schema_version": "quant_lab.factor_research_source_identity.v1",
            "windows": {
                dataset: [start.isoformat(), end.isoformat()]
                for dataset, (start, end) in sorted(windows.items())
            },
            "column_projections": FACTOR_RESEARCH_COLUMN_PROJECTIONS,
            "files": _alpha_source_identities(root, selected),
        }
    )


def _materialize_factor_research_snapshot_files(
    root: Path,
    temporary: Path,
    *,
    selected: list[tuple[str, Path, datetime | None, datetime | None]],
    windows: dict[str, tuple[datetime, datetime]],
    symbols: list[str],
    effective_registry: pl.DataFrame,
    trial_ledger: pl.DataFrame,
) -> list[ResearchDatasetReference]:
    references: list[ResearchDatasetReference] = []
    for ordinal, (dataset, indexed_source, _min_ts, _max_ts) in enumerate(selected):
        if indexed_source.is_symlink():
            raise ValueError("snapshot_source_path_escape")
        source = indexed_source.resolve(strict=True)
        if root not in source.parents:
            raise ValueError("snapshot_source_path_escape")
        source_relative = str(source.relative_to(root)).replace("\\", "/")
        before = source.stat()
        frame = _project_factor_research_source(
            source,
            dataset=dataset,
            window=windows[dataset],
            symbols=symbols,
        )
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("snapshot_source_changed_while_sealing")
        if frame.is_empty():
            continue
        part_id = model_content_sha256(
            {"source": source_relative, "ordinal": ordinal, "mtime_ns": before.st_mtime_ns}
        )[:16]
        relative_path = f"{dataset}/part-{part_id}.parquet"
        destination = temporary / "files" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(destination, compression="zstd")
        destination.chmod(0o440)
        min_ts, max_ts = _dataset_file_time_bounds(destination, dataset)
        stat = destination.stat()
        references.append(
            ResearchDatasetReference(
                dataset_name=dataset,
                source_relative_path=source_relative,
                relative_path=relative_path,
                sha256=sha256_file(destination),
                size_bytes=stat.st_size,
                row_count=frame.height,
                mtime_ns=before.st_mtime_ns,
                min_ts=min_ts,
                max_ts=max_ts,
            )
        )
    materialized_datasets = {item.dataset_name for item in references}
    required_data = {str(path).replace("\\", "/") for path in FACTOR_RESEARCH_DATA_DATASETS}
    missing_data = sorted(required_data - materialized_datasets)
    if missing_data:
        raise ValueError(f"factor_research_projected_dataset_empty:{','.join(missing_data)}")
    for dataset_path, frame, schema, name in (
        (
            RESEARCH_HYPOTHESIS_REGISTRY_DATASET,
            effective_registry,
            HYPOTHESIS_REGISTRY_SCHEMA,
            "authoritative-control",
        ),
        (RESEARCH_TRIAL_LEDGER_DATASET, trial_ledger, TRIAL_LEDGER_SCHEMA, "pre-registered"),
    ):
        dataset = str(dataset_path).replace("\\", "/")
        relative_path = f"{dataset}/part-{name}.parquet"
        destination = temporary / "files" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.select(list(schema)).write_parquet(destination, compression="zstd")
        destination.chmod(0o440)
        stat = destination.stat()
        references.append(
            ResearchDatasetReference(
                dataset_name=dataset,
                source_relative_path=relative_path,
                relative_path=relative_path,
                sha256=sha256_file(destination),
                size_bytes=stat.st_size,
                row_count=frame.height,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    references.sort(key=lambda item: (item.dataset_name, item.relative_path))
    return references


def _project_factor_research_source(
    source: Path,
    *,
    dataset: str,
    window: tuple[datetime, datetime],
    symbols: list[str],
) -> pl.DataFrame:
    schema = pl.read_parquet_schema(source)
    selected_columns = [
        column for column in FACTOR_RESEARCH_COLUMN_PROJECTIONS[dataset] if column in schema
    ]
    if not selected_columns:
        return pl.DataFrame()
    lazy = pl.scan_parquet(source)
    time_column = next(
        (column for column in SNAPSHOT_TIME_COLUMNS.get(dataset, ()) if column in schema),
        None,
    )
    if time_column is not None:
        start, end = window
        timestamp = _snapshot_timestamp_expr(time_column)
        lazy = lazy.filter((timestamp >= start) & (timestamp < end))
    if "symbol" in schema:
        lazy = lazy.filter(pl.col("symbol").cast(pl.Utf8).is_in(symbols))
    if dataset == "silver/market_bar":
        if "timeframe" in schema:
            lazy = lazy.filter(pl.col("timeframe").cast(pl.Utf8).str.to_lowercase() == "1h")
        if "is_closed" in schema:
            lazy = lazy.filter(pl.col("is_closed").cast(pl.Boolean, strict=False).fill_null(False))
    return lazy.select(selected_columns).collect(engine="streaming")


def _alpha_source_identities(
    root: Path,
    selected: list[tuple[str, Path, datetime | None, datetime | None]],
) -> list[dict[str, object]]:
    identities: list[dict[str, object]] = []
    for dataset, source, min_ts, max_ts in selected:
        stat = source.stat()
        identities.append(
            {
                "dataset": dataset,
                "path": str(source.relative_to(root)).replace("\\", "/"),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "min_ts": min_ts.isoformat() if min_ts is not None else None,
                "max_ts": max_ts.isoformat() if max_ts is not None else None,
            }
        )
    identities.sort(key=lambda row: (str(row["dataset"]), str(row["path"])))
    return identities


def _alpha_required_symbols(
    selected: list[tuple[str, Path, datetime | None, datetime | None]],
) -> list[str]:
    symbols = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"}
    symbol_datasets = {
        "gold/expanded_universe_quality",
        "gold/factor_value",
        "gold/strategy_evidence",
        "gold/strategy_evidence_sample",
    }
    for dataset, source, _min_ts, _max_ts in selected:
        if dataset not in symbol_datasets:
            continue
        try:
            schema = pl.read_parquet_schema(source)
            if "symbol" not in schema:
                continue
            lazy = pl.scan_parquet(source).select("symbol")
            frame = lazy.collect(engine="streaming")
        except (OSError, pl.exceptions.PolarsError):
            continue
        for value in frame.get_column("symbol").drop_nulls().unique().to_list():
            symbol = normalize_symbol(value)
            if symbol:
                symbols.add(symbol)
    return sorted(symbols)


def _materialize_alpha_factory_snapshot_files(
    root: Path,
    temporary: Path,
    *,
    selected: list[tuple[str, Path, datetime | None, datetime | None]],
    windows: dict[str, tuple[datetime, datetime]],
    symbols: list[str],
    effective_registry: pl.DataFrame,
    exact_dataset_days: dict[str, date] | None = None,
) -> list[ResearchDatasetReference]:
    references: list[ResearchDatasetReference] = []
    exact_days = exact_dataset_days or {}
    for ordinal, (dataset, indexed_source, _min_ts, _max_ts) in enumerate(selected):
        if indexed_source.is_symlink():
            raise ValueError("snapshot_source_path_escape")
        source = indexed_source.resolve(strict=True)
        if root not in source.parents:
            raise ValueError("snapshot_source_path_escape")
        source_relative = str(source.relative_to(root)).replace("\\", "/")
        before = source.stat()
        frame = _project_alpha_factory_source(
            source,
            dataset=dataset,
            window=windows[dataset],
            symbols=symbols,
            exact_day=exact_days.get(dataset),
        )
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("snapshot_source_changed_while_sealing")
        if frame.is_empty():
            continue
        part_id = model_content_sha256(
            {"source": source_relative, "ordinal": ordinal, "mtime_ns": before.st_mtime_ns}
        )[:16]
        relative_path = f"{dataset}/part-{part_id}.parquet"
        destination = temporary / "files" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(destination, compression="zstd")
        destination.chmod(0o440)
        min_ts, max_ts = _dataset_file_time_bounds(destination, dataset)
        stat = destination.stat()
        references.append(
            ResearchDatasetReference(
                dataset_name=dataset,
                source_relative_path=source_relative,
                relative_path=relative_path,
                sha256=sha256_file(destination),
                size_bytes=stat.st_size,
                row_count=frame.height,
                mtime_ns=before.st_mtime_ns,
                min_ts=min_ts,
                max_ts=max_ts,
            )
        )
    registry_dataset = str(ALPHA_FACTORY_TEMPLATE_REGISTRY_DATASET).replace("\\", "/")
    registry_relative = f"{registry_dataset}/part-effective-control.parquet"
    registry_path = temporary / "files" / registry_relative
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    effective_registry.select(list(TEMPLATE_REGISTRY_SCHEMA)).write_parquet(
        registry_path,
        compression="zstd",
    )
    registry_path.chmod(0o440)
    registry_stat = registry_path.stat()
    references.append(
        ResearchDatasetReference(
            dataset_name=registry_dataset,
            source_relative_path=registry_relative,
            relative_path=registry_relative,
            sha256=sha256_file(registry_path),
            size_bytes=registry_stat.st_size,
            row_count=effective_registry.height,
            mtime_ns=registry_stat.st_mtime_ns,
        )
    )
    references.sort(key=lambda item: (item.dataset_name, item.relative_path))
    return references


def _project_alpha_factory_source(
    source: Path,
    *,
    dataset: str,
    window: tuple[datetime, datetime],
    symbols: list[str],
    exact_day: date | None = None,
) -> pl.DataFrame:
    schema = pl.read_parquet_schema(source)
    selected_columns = [
        column for column in ALPHA_FACTORY_COLUMN_PROJECTIONS[dataset] if column in schema
    ]
    if not selected_columns:
        return pl.DataFrame()
    lazy = pl.scan_parquet(source)
    time_column = next(
        (column for column in SNAPSHOT_TIME_COLUMNS.get(dataset, ()) if column in schema),
        None,
    )
    if time_column is not None:
        start, end = window
        timestamp = _snapshot_timestamp_expr(time_column)
        lazy = lazy.filter((timestamp >= start) & (timestamp < end))
        if exact_day is not None:
            exact_start = datetime.combine(exact_day, time.min, tzinfo=UTC)
            exact_end = exact_start + timedelta(days=1)
            lazy = lazy.filter((timestamp >= exact_start) & (timestamp < exact_end))
    if dataset in {"silver/market_bar", "gold/factor_value"} and "symbol" in schema:
        lazy = lazy.filter(pl.col("symbol").cast(pl.Utf8).is_in(symbols))
    return lazy.select(selected_columns).collect(engine="streaming")


def _latest_alpha_dataset_day(
    selected: list[tuple[str, Path, datetime | None, datetime | None]],
    *,
    dataset: str,
    window: tuple[datetime, datetime],
) -> date | None:
    latest: datetime | None = None
    start, end = window
    for source_dataset, source, _min_ts, _max_ts in selected:
        if source_dataset != dataset:
            continue
        schema = pl.read_parquet_schema(source)
        time_column = next(
            (column for column in SNAPSHOT_TIME_COLUMNS.get(dataset, ()) if column in schema),
            None,
        )
        if time_column is None:
            continue
        timestamp = _snapshot_timestamp_expr(time_column)
        frame = (
            pl.scan_parquet(source)
            .filter((timestamp >= start) & (timestamp < end))
            .select(timestamp.max().alias("latest"))
            .collect(engine="streaming")
        )
        if frame.is_empty():
            continue
        value = frame.item(0, "latest")
        if isinstance(value, datetime) and (latest is None or value > latest):
            latest = value
    return latest.date() if latest is not None else None


def _restrict_alpha_dataset_to_day(
    selected: list[tuple[str, Path, datetime | None, datetime | None]],
    *,
    dataset: str,
    effective_day: date,
) -> list[tuple[str, Path, datetime | None, datetime | None]]:
    start = datetime.combine(effective_day, time.min, tzinfo=UTC)
    end = start + timedelta(days=1)
    restricted: list[tuple[str, Path, datetime | None, datetime | None]] = []
    for item in selected:
        source_dataset, _source, min_ts, max_ts = item
        if source_dataset != dataset:
            restricted.append(item)
            continue
        if min_ts is not None and max_ts is not None and (max_ts < start or min_ts >= end):
            continue
        restricted.append(item)
    return restricted


def _snapshot_timestamp_expr(column: str) -> pl.Expr:
    return pl.coalesce(
        [
            pl.col(column).cast(pl.Datetime(time_zone="UTC"), strict=False),
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .str.to_datetime(time_zone="UTC", strict=False),
        ]
    )


def _load_required_file_index(
    root: Path,
    required_datasets: tuple[Path, ...] = ENTRY_QUALITY_INPUT_DATASETS,
) -> pl.DataFrame:
    required_names = {str(path).replace("\\", "/") for path in required_datasets}
    # Refresh file membership every request so a previously indexed dataset cannot
    # hide newly arrived files. The index reuses unchanged file metadata.
    build_lake_file_index(root, sorted(required_names))
    index = read_parquet_dataset(root / LAKE_FILE_INDEX)
    if index.is_empty():
        return pl.DataFrame(
            schema={
                "dataset": pl.Utf8,
                "path": pl.Utf8,
                "min_ts": pl.Utf8,
                "max_ts": pl.Utf8,
            }
        )
    if not {"dataset", "path", "min_ts", "max_ts"}.issubset(set(index.columns)):
        raise RuntimeError("lake_file_index_missing_or_invalid")
    return index


def _select_indexed_files(
    root: Path,
    index: pl.DataFrame,
    windows: dict[str, tuple[datetime, datetime]],
) -> list[tuple[str, Path, datetime | None, datetime | None]]:
    selected: list[tuple[str, Path, datetime | None, datetime | None]] = []
    for row in index.to_dicts():
        dataset = str(row.get("dataset") or "")
        if dataset not in windows:
            continue
        min_ts = _as_utc(row.get("min_ts"))
        max_ts = _as_utc(row.get("max_ts"))
        since, before = windows[dataset]
        relative = str(row.get("path") or "")
        if not relative:
            continue
        indexed = root / relative
        if _path_has_symlink(root, indexed):
            raise ValueError("lake_file_index_path_escape")
        candidate = indexed.resolve(strict=True)
        dataset_root = (root / dataset).resolve(strict=True)
        if (
            root not in dataset_root.parents
            or dataset_root not in candidate.parents
            or candidate.suffix != ".parquet"
        ):
            raise ValueError("lake_file_index_path_escape")
        if min_ts is None or max_ts is None:
            min_ts, max_ts = _dataset_file_time_bounds(candidate, dataset)
        if min_ts is not None and max_ts is not None and not (max_ts >= since and min_ts < before):
            continue
        selected.append((dataset, candidate, min_ts, max_ts))
    selected.sort(key=lambda item: (item[0], str(item[1])))
    return selected


def _parquet_row_count(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item())


def _dataset_file_time_bounds(
    path: Path,
    dataset: str,
) -> tuple[datetime | None, datetime | None]:
    schema = pl.read_parquet_schema(path)
    column = next(
        (name for name in SNAPSHOT_TIME_COLUMNS.get(dataset, ()) if name in schema),
        None,
    )
    if column is None:
        return None, None
    expression = pl.coalesce(
        [
            pl.col(column).cast(pl.Datetime(time_zone="UTC"), strict=False),
            pl.col(column)
            .cast(pl.Utf8, strict=False)
            .str.to_datetime(time_zone="UTC", strict=False),
        ]
    )
    bounds = (
        pl.scan_parquet(path)
        .select(expression.min().alias("min_ts"), expression.max().alias("max_ts"))
        .collect(engine="streaming")
    )
    return _as_utc(bounds.item(0, "min_ts")), _as_utc(bounds.item(0, "max_ts"))


def _as_utc(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _make_snapshot_read_only(path: Path) -> None:
    for candidate in sorted(path.rglob("*"), reverse=True):
        try:
            candidate.chmod(0o440 if candidate.is_file() else 0o550)
        except OSError:
            pass
    try:
        path.chmod(0o550)
    except OSError:
        pass


def _make_snapshot_writable(path: Path) -> None:
    path.chmod(0o750)
    for candidate in path.rglob("*"):
        candidate.chmod(0o640 if candidate.is_file() else 0o750)


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


def _git_commit() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    if len(value) != 40:
        raise RuntimeError("quant_lab_commit_unavailable")
    return value
