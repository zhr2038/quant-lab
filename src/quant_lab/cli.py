import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from quant_lab.contracts.models import AlphaResearchSpec
from quant_lab.costs.calibrate import calibrate_costs_for_day
from quant_lab.costs.health import read_cost_health_daily
from quant_lab.data.lake import (
    compact_parquet_dataset,
    compact_parquet_directory_files,
    repair_parquet_partition_values,
)
from quant_lab.e2e import run_v5_contract_e2e
from quant_lab.export.daily import export_daily_pack, validate_expert_pack
from quant_lab.features.publish import feature_health
from quant_lab.features.publish import publish_features as publish_feature_values
from quant_lab.gates.defaults import conservative_example_gate_decision
from quant_lab.ingest.okx_public import (
    MARKET_BAR_DATASET,
    OKXPublicClient,
    backfill_expanded_usdt_spot_market_bars,
    normalize_okx_candles_to_market_bars,
    publish_market_bars_to_lake,
)
from quant_lab.ingest.okx_readonly_private import (
    OKXReadOnlyClient,
    OKXReadOnlyConfig,
    publish_okx_bills_to_lake,
    publish_okx_fills_to_lake,
    publish_okx_orders_to_lake,
)
from quant_lab.ingest.okx_ws_public import collect_okx_public_ws, collect_okx_public_ws_universe
from quant_lab.ingest.v5_reports import inspect_v5_reports, publish_v5_reports_to_lake
from quant_lab.jobs.compact_market_data import build_market_data_1m_rollups
from quant_lab.ops.data_quality import run_data_quality
from quant_lab.ops.lake_health import (
    lake_dataset_quality_summary,
    lake_file_health_summary,
    write_lake_file_health_daily,
)
from quant_lab.ops.metrics import api_metrics_summary, job_run_summary, run_with_job_metrics
from quant_lab.ops.retention import prune_quant_lab_storage
from quant_lab.reports.enforce_readiness import write_enforce_readiness_report
from quant_lab.research.alpha_discovery import build_and_publish_alpha_discovery_board
from quant_lab.research.alpha_factory import build_and_publish_alpha_factory
from quant_lab.research.bnb_swing_exit_policy import (
    build_and_publish_bnb_swing_exit_policy_review,
)
from quant_lab.research.bootstrap_gold import bootstrap_gold_health
from quant_lab.research.btc_probe_exit_policy import (
    build_and_publish_btc_probe_exit_policy_review,
)
from quant_lab.research.candidate_labels import build_and_publish_candidate_labels
from quant_lab.research.diagnostics_refresh import refresh_research_diagnostics
from quant_lab.research.entry_quality import (
    build_and_publish_entry_quality,
    build_and_publish_entry_quality_history,
)
from quant_lab.research.expanded_universe import (
    build_and_publish_expanded_crypto_universe_shadow,
)
from quant_lab.research.paper_tracking import build_and_publish_paper_strategy_tracking
from quant_lab.research.portfolio import build_and_publish_research_portfolio_status
from quant_lab.research.publish import (
    build_and_publish_alpha_evidence,
    publish_gate_decisions_from_evidence,
    research_health,
)
from quant_lab.research.regime_router import build_and_publish_regime_router
from quant_lab.research.second_stage_alpha_factory import (
    build_and_publish_second_stage_alpha_factory,
)
from quant_lab.research.sol_protect_paper_loss import (
    build_and_publish_sol_protect_paper_loss_attribution,
)
from quant_lab.research.strategy_evidence import build_and_publish_strategy_evidence
from quant_lab.risk.publish import publish_risk_permission as publish_risk_permission_to_lake
from quant_lab.strategy_telemetry.analyze import analyze_v5_telemetry
from quant_lab.strategy_telemetry.bundle import safe_extract_v5_bundle, validate_v5_bundle
from quant_lab.strategy_telemetry.config import load_v5_telemetry_remote_config
from quant_lab.strategy_telemetry.ingest import ingest_v5_bundle as ingest_v5_bundle_file
from quant_lab.strategy_telemetry.ingest import ingest_v5_inbox as ingest_v5_inbox_dir
from quant_lab.strategy_telemetry.models import BundleLimits
from quant_lab.strategy_telemetry.remote_pull import RemoteBundlePuller
from quant_lab.strategy_telemetry.sanitize import scan_for_secrets

app = typer.Typer(help="quant-lab read-only research utilities.")

ReportsDirArgument = Annotated[
    Path,
    typer.Argument(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="V5 reports directory to inspect without mutating it.",
    ),
]


@app.command("gate-example")
def gate_example() -> None:
    decision = conservative_example_gate_decision()
    typer.echo(decision.model_dump_json(indent=2))


@app.command("inspect-v5")
def inspect_v5(reports_dir: ReportsDirArgument) -> None:
    inspection = inspect_v5_reports(reports_dir)
    typer.echo(inspection.model_dump_json(indent=2))


@app.command("publish-v5")
def publish_v5(
    reports_dir: ReportsDirArgument,
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root to publish Parquet datasets into.",
        ),
    ],
) -> None:
    result = publish_v5_reports_to_lake(reports_dir, lake_root)
    typer.echo(result.model_dump_json(indent=2))


@app.command("okx-fetch-candles")
def okx_fetch_candles(
    inst_id: Annotated[str, typer.Option("--inst-id", help="OKX instrument ID.")],
    bar: Annotated[str, typer.Option("--bar", help="OKX candle bar, for example 1H.")],
    market_type: Annotated[
        str,
        typer.Option("--market-type", help="Market type, for example SPOT."),
    ],
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root to publish market bars into.",
        ),
    ],
    limit: Annotated[int, typer.Option("--limit", min=1, max=300)] = 100,
    after: Annotated[str | None, typer.Option("--after")] = None,
    before: Annotated[str | None, typer.Option("--before")] = None,
    history: Annotated[bool, typer.Option("--history")] = False,
) -> None:
    client = OKXPublicClient()
    candles = (
        client.get_history_candles(inst_id, bar, after=after, before=before, limit=limit)
        if history
        else client.get_candles(inst_id, bar, after=after, before=before, limit=limit)
    )
    market_bars = normalize_okx_candles_to_market_bars(
        candles=candles,
        inst_id=inst_id,
        bar=bar,
        market_type=market_type,
    )
    total_rows = publish_market_bars_to_lake(market_bars, lake_root)
    typer.echo(
        json.dumps(
            {
                "source": "okx_public_rest",
                "inst_id": inst_id,
                "bar": bar,
                "market_type": market_type,
                "fetched_candles": len(candles),
                "published_market_bars": len(market_bars),
                "market_bar_rows": total_rows,
                "dataset_path": str(lake_root / MARKET_BAR_DATASET),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("okx-backfill-expanded-universe")
def okx_backfill_expanded_universe(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root to publish expanded OKX spot market bars into.",
        ),
    ],
    bar: Annotated[str, typer.Option("--bar", help="OKX candle bar.")] = "1H",
    max_symbols: Annotated[int, typer.Option("--max-symbols", min=1, max=100)] = 30,
    history_pages: Annotated[int, typer.Option("--history-pages", min=1, max=20)] = 8,
    limit: Annotated[int, typer.Option("--limit", min=1, max=300)] = 100,
    min_quote_volume_24h: Annotated[
        float,
        typer.Option("--min-quote-volume-24h", min=0.0),
    ] = 1_000_000.0,
    max_spread_bps: Annotated[float, typer.Option("--max-spread-bps", min=0.0)] = 20.0,
    min_price: Annotated[float, typer.Option("--min-price", min=0.0)] = 0.01,
    blacklist: Annotated[
        str | None,
        typer.Option("--blacklist", help="Comma-separated symbols to exclude."),
    ] = None,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="okx-backfill-expanded-universe",
        func=lambda: backfill_expanded_usdt_spot_market_bars(
            lake_root=lake_root,
            bar=bar,
            max_symbols=max_symbols,
            history_pages=history_pages,
            limit=limit,
            min_quote_volume_24h=min_quote_volume_24h,
            max_spread_bps=max_spread_bps,
            min_price=min_price,
            blacklist=[item.strip() for item in (blacklist or "").split(",") if item.strip()],
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("okx-ws-run")
def okx_ws_run(
    inst_id: Annotated[str, typer.Option("--inst-id", help="OKX instrument ID.")],
    channels: Annotated[
        str,
        typer.Option("--channels", help="Comma-separated public channels."),
    ],
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root to publish realtime public data into.",
        ),
    ],
    market_type: Annotated[
        str,
        typer.Option("--market-type", help="Market type used for candle normalization."),
    ] = "SPOT",
    max_messages: Annotated[
        int | None,
        typer.Option("--max-messages", min=1, help="Stop after this many messages."),
    ] = None,
) -> None:
    parsed_channels = [channel.strip() for channel in channels.split(",") if channel.strip()]
    summary = asyncio.run(
        collect_okx_public_ws(
            inst_id=inst_id,
            channels=parsed_channels,
            lake_root=lake_root,
            market_type=market_type,
            max_messages=max_messages,
        )
    )
    typer.echo(summary.model_dump_json(indent=2))


@app.command("okx-ws-collect-universe")
def okx_ws_collect_universe(
    symbols: Annotated[
        str,
        typer.Option("--symbols", help="Comma-separated OKX instrument IDs."),
    ],
    channels: Annotated[
        str,
        typer.Option("--channels", help="Comma-separated public channels."),
    ],
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root to publish realtime public data into.",
        ),
    ],
    market_type: Annotated[str, typer.Option("--market-type")] = "SPOT",
    flush_interval_seconds: Annotated[
        float,
        typer.Option("--flush-interval-seconds", min=0.1),
    ] = 60.0,
    flush_max_messages: Annotated[
        int,
        typer.Option("--flush-max-messages", min=1),
    ] = 10_000,
    max_messages: Annotated[
        int | None,
        typer.Option("--max-messages", min=1, hidden=True),
    ] = None,
) -> None:
    parsed_symbols = [symbol.strip() for symbol in symbols.split(",") if symbol.strip()]
    parsed_channels = [channel.strip() for channel in channels.split(",") if channel.strip()]
    summary = asyncio.run(
        collect_okx_public_ws_universe(
            symbols=parsed_symbols,
            channels=parsed_channels,
            lake_root=lake_root,
            market_type=market_type,
            flush_interval_seconds=flush_interval_seconds,
            flush_max_messages=flush_max_messages,
            max_messages=max_messages,
        )
    )
    typer.echo(summary.model_dump_json(indent=2))


@app.command("okx-fetch-fills")
def okx_fetch_fills(
    inst_type: Annotated[str, typer.Option("--inst-type", help="OKX instrument type.")],
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root to publish read-only private fills into.",
        ),
    ],
    inst_id: Annotated[
        str | None,
        typer.Option("--inst-id", help="Optional instrument ID."),
    ] = None,
    after: Annotated[str | None, typer.Option("--after")] = None,
    before: Annotated[str | None, typer.Option("--before")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 100,
) -> None:
    client = OKXReadOnlyClient(OKXReadOnlyConfig.from_env())
    fills = client.get_fills_history(
        inst_type=inst_type,
        inst_id=inst_id,
        after=after,
        before=before,
        limit=limit,
    )
    publish_result = publish_okx_fills_to_lake(fills, lake_root)
    typer.echo(
        json.dumps(
            {
                "source": "okx_readonly_private",
                "inst_type": inst_type,
                "inst_id": inst_id,
                "fetched_fills": len(fills),
                **publish_result,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("okx-fetch-bills")
def okx_fetch_bills(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root to publish read-only private bills into.",
        ),
    ],
    ccy: Annotated[str | None, typer.Option("--ccy", help="Optional currency.")] = None,
    after: Annotated[str | None, typer.Option("--after")] = None,
    before: Annotated[str | None, typer.Option("--before")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 100,
) -> None:
    client = OKXReadOnlyClient(OKXReadOnlyConfig.from_env())
    bills = client.get_account_bills(ccy=ccy, after=after, before=before, limit=limit)
    publish_result = publish_okx_bills_to_lake(bills, lake_root)
    typer.echo(
        json.dumps(
            {
                "source": "okx_readonly_private",
                "ccy": ccy,
                "fetched_bills": len(bills),
                **publish_result,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("okx-backfill-readonly")
def okx_backfill_readonly(
    inst_type: Annotated[str, typer.Option("--inst-type", help="OKX instrument type.")],
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root to publish read-only private datasets into.",
        ),
    ],
    inst_id: Annotated[str | None, typer.Option("--inst-id")] = None,
    ccy: Annotated[str | None, typer.Option("--ccy")] = None,
    begin: Annotated[str | None, typer.Option("--begin")] = None,
    end: Annotated[str | None, typer.Option("--end")] = None,
    max_pages: Annotated[int, typer.Option("--max-pages", min=1)] = 20,
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 100,
) -> None:
    client = OKXReadOnlyClient(
        OKXReadOnlyConfig.from_env().model_copy(update={"max_pages": max_pages})
    )
    fills = client.backfill_fills_history(
        inst_type=inst_type,
        inst_id=inst_id,
        begin=begin,
        end=end,
        limit=limit,
        max_pages=max_pages,
    )
    bills = client.backfill_account_bills(
        ccy=ccy,
        begin=begin,
        end=end,
        limit=limit,
        max_pages=max_pages,
    )
    orders = client.backfill_orders_history(
        inst_type=inst_type,
        inst_id=inst_id,
        begin=begin,
        end=end,
        limit=limit,
        max_pages=max_pages,
    )
    fill_result = publish_okx_fills_to_lake(fills, lake_root)
    bill_result = publish_okx_bills_to_lake(bills, lake_root)
    order_result = publish_okx_orders_to_lake(orders, lake_root)
    typer.echo(
        json.dumps(
            {
                "source": "okx_readonly_private",
                "inst_type": inst_type,
                "inst_id": inst_id,
                "ccy": ccy,
                "fetched_fills": len(fills),
                "fetched_bills": len(bills),
                "fetched_orders": len(orders),
                **fill_result,
                **bill_result,
                **order_result,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("calibrate-costs")
def calibrate_costs(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root containing silver OKX datasets.",
        ),
    ],
    day: Annotated[str, typer.Option("--day", help="UTC day in YYYY-MM-DD format.")],
    min_sample_count: Annotated[int, typer.Option("--min-sample-count", min=1)] = 30,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="calibrate-costs",
        func=lambda: calibrate_costs_for_day(
            lake_root=lake_root,
            day=day,
            min_sample_count=min_sample_count,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("cost-health")
def cost_health_command(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            help="quant-lab lake root containing gold cost_health_daily.",
        ),
    ],
    day: Annotated[str | None, typer.Option("--day")] = None,
) -> None:
    typer.echo(json.dumps(read_cost_health_daily(lake_root, day=day), indent=2, sort_keys=True))


@app.command("build-expanded-universe-shadow")
def build_expanded_universe_shadow_command(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root containing market/research datasets.",
        ),
    ],
    as_of_date: Annotated[str | None, typer.Option("--as-of-date")] = None,
    max_candidates: Annotated[int, typer.Option("--max-candidates", min=1, max=100)] = 30,
    min_quote_volume_24h: Annotated[
        float,
        typer.Option("--min-quote-volume-24h", min=0.0),
    ] = 1_000_000.0,
    max_spread_bps: Annotated[float, typer.Option("--max-spread-bps", min=0.0)] = 20.0,
    min_coverage_bars: Annotated[int, typer.Option("--min-coverage-bars", min=1)] = 24 * 30,
    blacklist: Annotated[
        str | None,
        typer.Option("--blacklist", help="Comma-separated symbols to exclude."),
    ] = None,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-expanded-universe-shadow",
        func=lambda: build_and_publish_expanded_crypto_universe_shadow(
            lake_root,
            as_of_date=as_of_date,
            max_candidates=max_candidates,
            min_quote_volume_24h=min_quote_volume_24h,
            max_spread_bps=max_spread_bps,
            min_coverage_bars=min_coverage_bars,
            blacklist=[item.strip() for item in (blacklist or "").split(",") if item.strip()],
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("lake-health")
def lake_health_command(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            help="quant-lab lake root to inspect.",
        ),
    ],
    compact_output: Annotated[
        bool,
        typer.Option(
            "--compact-output/--full-output",
            help="Emit a single-line summary suitable for systemd journals.",
        ),
    ] = False,
    include_quality: Annotated[
        bool,
        typer.Option(
            "--include-quality/--file-health-only",
            help="Also run registry-driven schema/freshness/key quality checks.",
        ),
    ] = False,
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            help="Optional comma-separated dataset names to inspect.",
        ),
    ] = None,
) -> None:
    result = write_lake_file_health_daily(lake_root)
    dataset_names = _parse_dataset_names(dataset)
    if include_quality:
        result["data_quality"] = lake_dataset_quality_summary(
            lake_root,
            dataset_names=dataset_names,
            include_checks=not compact_output,
        )
    if compact_output:
        typer.echo(json.dumps(_compact_lake_health_payload(result), sort_keys=True, default=str))
    else:
        typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


@app.command("data-quality")
def data_quality_command(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            help="quant-lab lake root to inspect.",
        ),
    ],
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            help="Optional comma-separated dataset names to inspect.",
        ),
    ] = None,
    compact_output: Annotated[
        bool,
        typer.Option(
            "--compact-output/--full-output",
            help="Emit a compact data-quality summary with top failing checks.",
        ),
    ] = False,
) -> None:
    result = run_data_quality(
        lake_root,
        dataset_names=_parse_dataset_names(dataset),
    ).to_dict(include_checks=True)
    output = _compact_data_quality_payload(result) if compact_output else result
    typer.echo(
        json.dumps(
            output,
            indent=None if compact_output else 2,
            sort_keys=True,
            default=str,
        )
    )


def _compact_lake_health_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    health_rows = rows if isinstance(rows, list) else []
    warning_rows = [
        row
        for row in health_rows
        if isinstance(row, dict) and str(row.get("status") or "OK") != "OK"
    ]
    largest_file_rows = sorted(
        [row for row in health_rows if isinstance(row, dict)],
        key=lambda row: int(row.get("parquet_file_count") or 0),
        reverse=True,
    )[:5]
    return {
        "dataset_count": payload.get("dataset_count", len(health_rows)),
        "total_parquet_files": payload.get("total_parquet_files", 0),
        "warning_count": payload.get("warning_count", len(warning_rows)),
        "data_quality": _compact_data_quality_payload(
            payload.get("data_quality") if isinstance(payload.get("data_quality"), dict) else {}
        ),
        "warnings": [
            {
                "dataset": row.get("dataset"),
                "status": row.get("status"),
                "warning": row.get("warning"),
                "parquet_file_count": row.get("parquet_file_count"),
            }
            for row in warning_rows[:10]
        ],
        "top_file_count_datasets": [
            {
                "dataset": row.get("dataset"),
                "parquet_file_count": row.get("parquet_file_count"),
                "partition_dir_count": row.get("partition_dir_count"),
                "status": row.get("status"),
            }
            for row in largest_file_rows
        ],
    }


@app.command("compact-lake-dataset")
def compact_lake_dataset_command(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root containing the dataset.",
        ),
    ],
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            help=(
                "Dataset name: okx_public_ws, trade_print, orderbook_snapshot, "
                "or a relative path."
            ),
        ),
    ],
    target_rows_per_file: Annotated[
        int,
        typer.Option("--target-rows-per-file", min=1),
    ] = 250_000,
    max_source_files_per_batch: Annotated[
        int,
        typer.Option("--max-source-files-per-batch", min=1),
    ] = 5_000,
    max_source_batch_bytes: Annotated[
        int,
        typer.Option(
            "--max-source-batch-bytes",
            min=0,
            help=(
                "Maximum source Parquet bytes to read per compaction batch. "
                "Use 0 to read QUANT_LAB_COMPACT_MAX_SOURCE_BATCH_BYTES."
            ),
        ),
    ] = 0,
    direct_only: Annotated[
        bool,
        typer.Option(
            "--direct-only/--recursive",
            help="Compact only Parquet files directly in the dataset directory.",
        ),
    ] = False,
    include_existing_compact_files: Annotated[
        bool,
        typer.Option(
            "--include-existing-compact-files/--skip-existing-compact-files",
            help=(
                "When using --direct-only, also consolidate existing compact_*.parquet "
                "outputs. Default skips them to avoid repeatedly reading historical data."
            ),
        ),
    ] = False,
    compact_output: Annotated[
        bool,
        typer.Option(
            "--compact-output/--full-output",
            help="Emit a single-line summary suitable for systemd journals.",
        ),
    ] = False,
) -> None:
    dataset_path, partition_by = _compact_dataset_target(lake_root, dataset)
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name=f"compact-lake-dataset:{dataset}",
        func=lambda: (
            compact_parquet_directory_files(
                dataset_path,
                target_rows_per_file=target_rows_per_file,
                max_source_files_per_batch=max_source_files_per_batch,
                max_source_batch_bytes=max_source_batch_bytes,
                include_existing_compact_files=include_existing_compact_files,
            )
            if direct_only
            else compact_parquet_dataset(
                dataset_path,
                partition_by=partition_by,
                target_rows_per_file=target_rows_per_file,
                max_source_files_per_batch=max_source_files_per_batch,
                max_source_batch_bytes=max_source_batch_bytes,
            )
        ),
    )
    typer.echo(json.dumps(result.__dict__, indent=None if compact_output else 2, sort_keys=True))


@app.command("build-market-data-rollups")
def build_market_data_rollups_command(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root containing silver trade/orderbook source datasets.",
        ),
    ],
    apply: Annotated[
        bool,
        typer.Option(
            "--apply/--dry-run",
            help="Write derived 1m rollups. Default is dry-run.",
        ),
    ] = False,
    compact_output: Annotated[
        bool,
        typer.Option(
            "--compact-output/--full-output",
            help="Emit a single-line summary suitable for systemd journals.",
        ),
    ] = False,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-market-data-rollups",
        func=lambda: build_market_data_1m_rollups(lake_root, dry_run=not apply),
    )
    typer.echo(json.dumps(result.to_dict(), indent=None if compact_output else 2, sort_keys=True))


@app.command("repair-lake-partitions")
def repair_lake_partitions_command(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root containing the dataset.",
        ),
    ],
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            help=(
                "Dataset name: okx_public_ws, trade_print, orderbook_snapshot, "
                "or a relative path."
            ),
        ),
    ],
    target_rows_per_file: Annotated[
        int,
        typer.Option("--target-rows-per-file", min=1),
    ] = 250_000,
    max_source_files_per_batch: Annotated[
        int,
        typer.Option("--max-source-files-per-batch", min=1),
    ] = 5_000,
    max_source_batch_bytes: Annotated[
        int,
        typer.Option(
            "--max-source-batch-bytes",
            min=0,
            help=(
                "Maximum source Parquet bytes to read per repair batch. "
                "Use 0 to read QUANT_LAB_COMPACT_MAX_SOURCE_BATCH_BYTES."
            ),
        ),
    ] = 0,
    compact_output: Annotated[
        bool,
        typer.Option(
            "--compact-output/--full-output",
            help="Emit a single-line summary suitable for systemd journals.",
        ),
    ] = False,
) -> None:
    dataset_path, partition_by = _compact_dataset_target(lake_root, dataset)
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name=f"repair-lake-partitions:{dataset}",
        func=lambda: repair_parquet_partition_values(
            dataset_path,
            partition_by=partition_by,
            target_rows_per_file=target_rows_per_file,
            max_source_files_per_batch=max_source_files_per_batch,
            max_source_batch_bytes=max_source_batch_bytes,
        ),
    )
    typer.echo(json.dumps(result.__dict__, indent=None if compact_output else 2, sort_keys=True))


@app.command("ops-summary")
def ops_summary_command(
    lake_root: Annotated[
        Path,
        typer.Option("--lake-root", file_okay=False, dir_okay=True),
    ],
    day: Annotated[
        str | None,
        typer.Option(
            "--day",
            help=(
                "Metrics day to summarize. Combine with --since-minutes 0 for a full day; "
                "pass 'all' for full history."
            ),
        ),
    ] = None,
    since_minutes: Annotated[
        int,
        typer.Option(
            "--since-minutes",
            min=0,
            help="Recent metrics window. Defaults to 60 minutes; set 0 to disable.",
        ),
    ] = 60,
    compact_output: Annotated[
        bool,
        typer.Option(
            "--compact-output/--full-output",
            help="Emit a compact operational summary without large nested tables.",
        ),
    ] = False,
    include_quality: Annotated[
        bool,
        typer.Option(
            "--include-quality/--skip-quality",
            help="Include registry-driven dataset quality checks.",
        ),
    ] = False,
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            help="Optional comma-separated dataset names for --include-quality.",
        ),
    ] = None,
) -> None:
    all_history = str(day or "").strip().lower() == "all"
    summary_day = None if all_history else day
    summary_since_minutes = None if all_history or since_minutes <= 0 else since_minutes
    payload = {
        "api_metrics": api_metrics_summary(
            lake_root,
            day=summary_day,
            since_minutes=summary_since_minutes,
        ),
        "job_runs": job_run_summary(
            lake_root,
            day=summary_day,
            since_minutes=summary_since_minutes,
        ),
        "lake_file_health": lake_file_health_summary(lake_root),
    }
    if include_quality:
        payload["data_quality"] = lake_dataset_quality_summary(
            lake_root,
            dataset_names=_parse_dataset_names(dataset),
            include_checks=not compact_output,
        )
    output = _compact_ops_summary_payload(payload) if compact_output else payload
    typer.echo(
        json.dumps(
            output,
            indent=None if compact_output else 2,
            sort_keys=True,
            default=str,
        )
    )


def _compact_ops_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    api_metrics = payload.get("api_metrics") if isinstance(payload.get("api_metrics"), dict) else {}
    job_runs = payload.get("job_runs") if isinstance(payload.get("job_runs"), dict) else {}
    lake_health = (
        payload.get("lake_file_health")
        if isinstance(payload.get("lake_file_health"), dict)
        else {}
    )
    return {
        "api_metrics": _compact_api_metrics_payload(api_metrics),
        "job_runs": _compact_job_run_payload(job_runs),
        "lake_file_health": _compact_lake_health_payload(lake_health),
        "data_quality": _compact_data_quality_payload(
            payload.get("data_quality") if isinstance(payload.get("data_quality"), dict) else {}
        ),
    }


def _compact_data_quality_payload(data_quality: dict[str, Any]) -> dict[str, Any]:
    if not data_quality:
        return {}
    checks = data_quality.get("checks") if isinstance(data_quality.get("checks"), list) else []
    failing = [
        check
        for check in checks
        if isinstance(check, dict) and str(check.get("status") or "PASS") != "PASS"
    ]
    failing = sorted(
        failing,
        key=lambda check: (
            1 if str(check.get("rule") or "") == "freshness" else 0,
            str(check.get("status") or ""),
            str(check.get("dataset") or ""),
            str(check.get("rule") or ""),
        ),
    )
    return {
        "status": data_quality.get("status"),
        "dataset_count": data_quality.get("dataset_count"),
        "check_count": data_quality.get("check_count"),
        "fail_count": data_quality.get("fail_count"),
        "warning_count": data_quality.get("warning_count"),
        "failing_checks": [
            {
                "dataset": check.get("dataset"),
                "rule": check.get("rule"),
                "status": check.get("status"),
                "severity": check.get("severity"),
                "detail": check.get("detail"),
                "next_action": check.get("next_action"),
            }
            for check in failing[:20]
        ],
    }


def _parse_dataset_names(dataset: str | None) -> list[str] | None:
    if dataset is None:
        return None
    parsed = [item.strip() for item in dataset.split(",") if item.strip()]
    return parsed or None


def _compact_api_metrics_payload(api_metrics: dict[str, Any]) -> dict[str, Any]:
    by_path = api_metrics.get("by_path") if isinstance(api_metrics.get("by_path"), dict) else {}
    top_paths = sorted(
        (
            {"path": str(path), "count": int(count or 0)}
            for path, count in by_path.items()
        ),
        key=lambda row: row["count"],
        reverse=True,
    )[:10]
    slow_paths = api_metrics.get("slow_paths")
    return {
        "request_count": int(api_metrics.get("request_count") or 0),
        "by_status_code": api_metrics.get("by_status_code") or {},
        "latency_ms": api_metrics.get("latency_ms") or {},
        "top_paths": top_paths,
        "slow_paths": slow_paths[:10] if isinstance(slow_paths, list) else [],
    }


def _compact_job_run_payload(job_runs: dict[str, Any]) -> dict[str, Any]:
    jobs = job_runs.get("jobs")
    job_rows = jobs if isinstance(jobs, list) else []
    historical_failed_jobs = [
        row
        for row in job_rows
        if isinstance(row, dict) and int(row.get("failure_count") or 0) > 0
    ]
    current_failed_jobs = [
        row
        for row in job_rows
        if isinstance(row, dict) and str(row.get("latest_status") or "").lower() == "failed"
    ]
    slow_jobs = sorted(
        (row for row in job_rows if isinstance(row, dict)),
        key=lambda row: (
            float(row.get("p95_s") or 0.0),
            float(row.get("max_s") or 0.0),
            int(row.get("run_count") or 0),
        ),
        reverse=True,
    )[:20]
    return {
        "run_count": int(job_runs.get("run_count") or 0),
        "job_count": len(job_rows),
        "failed_job_count": len(current_failed_jobs),
        "historical_failed_job_count": len(historical_failed_jobs),
        "failed_jobs": _compact_job_rows(current_failed_jobs[:20]),
        "historical_failed_jobs": _compact_job_rows(historical_failed_jobs[:20]),
        "slow_jobs": _compact_job_rows(slow_jobs),
    }


def _compact_job_rows(rows: list[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        output.append(
            {
                "job_name": row.get("job_name"),
                "latest_status": row.get("latest_status"),
                "run_count": row.get("run_count"),
                "failure_count": row.get("failure_count"),
                "latest_duration_s": row.get("latest_duration_s"),
                "p95_s": row.get("p95_s"),
                "max_s": row.get("max_s"),
                "latest_finished_at": row.get("latest_finished_at"),
            }
        )
    return output


@app.command("prune-storage-retention")
def prune_storage_retention_command(
    base_dir: Annotated[
        Path,
        typer.Option(
            "--base-dir",
            file_okay=False,
            dir_okay=True,
            help="quant-lab production data root, for example /var/lib/quant-lab.",
        ),
    ] = Path("/var/lib/quant-lab"),
    keep_redacted_archive_days: Annotated[
        int,
        typer.Option("--keep-redacted-archive-days", min=1),
    ] = 3,
    keep_inbox_days: Annotated[int, typer.Option("--keep-inbox-days", min=1)] = 2,
    keep_export_packs: Annotated[int, typer.Option("--keep-export-packs", min=1)] = 5,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--apply",
            help="Preview removals by default; pass --apply to remove files.",
        ),
    ] = True,
    max_removed_paths_reported: Annotated[
        int,
        typer.Option("--max-removed-paths-reported", min=0),
    ] = 50,
) -> None:
    result = prune_quant_lab_storage(
        base_dir=base_dir,
        keep_redacted_archive_days=keep_redacted_archive_days,
        keep_inbox_days=keep_inbox_days,
        keep_export_packs=keep_export_packs,
        dry_run=dry_run,
    )
    typer.echo(
        json.dumps(
            result.to_dict(max_removed_paths_reported=max_removed_paths_reported),
            indent=2,
            sort_keys=True,
        )
    )


@app.command("bootstrap-gold-health")
def bootstrap_gold_health_command(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root to fill conservative gold health datasets.",
        ),
    ],
    strategy: Annotated[str, typer.Option("--strategy")] = "v5",
    version: Annotated[str, typer.Option("--version")] = "bootstrap",
    day: Annotated[str, typer.Option("--day", help="UTC day YYYY-MM-DD or auto.")] = "auto",
) -> None:
    result = bootstrap_gold_health(
        lake_root=lake_root,
        strategy=strategy,
        version=version,
        day=day,
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("publish-features")
def publish_features(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="quant-lab lake root containing silver market_bar.",
        ),
    ],
    feature_set: Annotated[str, typer.Option("--feature-set")] = "core",
    feature_version: Annotated[str, typer.Option("--feature-version")] = "v0.1",
    timeframe: Annotated[str, typer.Option("--timeframe")] = "1H",
    symbols: Annotated[str | None, typer.Option("--symbols")] = None,
    start: Annotated[datetime | None, typer.Option("--start")] = None,
    end: Annotated[datetime | None, typer.Option("--end")] = None,
    drop_null: Annotated[bool, typer.Option("--drop-null/--keep-null")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    allow_schema_replace: Annotated[
        bool,
        typer.Option(
            "--allow-schema-replace/--no-schema-replace",
            help="Replace an incompatible existing feature dataset after manual review.",
        ),
    ] = False,
) -> None:
    parsed_symbols = (
        [symbol.strip() for symbol in symbols.split(",") if symbol.strip()] if symbols else None
    )
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="publish-features",
        func=lambda: publish_feature_values(
            lake_root=lake_root,
            feature_set=feature_set,
            feature_version=feature_version,
            timeframe=timeframe,
            symbols=parsed_symbols,
            start=start,
            end=end,
            drop_null=drop_null,
            dry_run=dry_run,
            allow_schema_replace=allow_schema_replace,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("feature-health")
def feature_health_command(
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            help="quant-lab lake root containing gold feature quality datasets.",
        ),
    ],
    feature_set: Annotated[str, typer.Option("--feature-set")] = "core",
    date: Annotated[str | None, typer.Option("--date")] = None,
) -> None:
    result = feature_health(lake_root=lake_root, feature_set=feature_set, date=date)
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-alpha-evidence")
def build_alpha_evidence_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    alpha_id: Annotated[str, typer.Option("--alpha-id")] = "v5.core.momentum",
    version: Annotated[str, typer.Option("--version")] = "v0.1",
    feature_set: Annotated[str, typer.Option("--feature-set")] = "core",
    feature_version: Annotated[str, typer.Option("--feature-version")] = "v0.1",
    feature_names: Annotated[str, typer.Option("--feature-names")] = "close_return_24",
    timeframe: Annotated[str, typer.Option("--timeframe")] = "1H",
    label_horizon_bars: Annotated[int, typer.Option("--label-horizon-bars", min=1)] = 4,
    decision_delay_bars: Annotated[int, typer.Option("--decision-delay-bars", min=1)] = 1,
    universe_id: Annotated[str, typer.Option("--universe-id")] = "okx-major-spot",
    strategy: Annotated[str, typer.Option("--strategy")] = "v5",
    cost_quantile: Annotated[str, typer.Option("--cost-quantile")] = "p75",
    min_samples: Annotated[int, typer.Option("--min-samples", min=1)] = 100,
) -> None:
    spec = AlphaResearchSpec(
        alpha_id=alpha_id,
        version=version,
        feature_set=feature_set,
        feature_version=feature_version,
        feature_names=[name.strip() for name in feature_names.split(",") if name.strip()],
        timeframe=timeframe,
        label_horizon_bars=label_horizon_bars,
        decision_delay_bars=decision_delay_bars,
        universe_id=universe_id,
        strategy=strategy,
        cost_quantile=cost_quantile,
        min_samples=min_samples,
    )
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-alpha-evidence",
        func=lambda: build_and_publish_alpha_evidence(lake_root=lake_root, spec=spec),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-strategy-evidence")
def build_strategy_evidence_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
    min_live_samples: Annotated[int, typer.Option("--min-live-samples", min=30)] = 30,
    mode: Annotated[str, typer.Option("--mode", help="full or incremental.")] = "incremental",
    lookback_days: Annotated[int, typer.Option("--lookback-days", min=1)] = 8,
    include_historical_outcomes: Annotated[
        bool,
        typer.Option("--include-historical-outcomes/--skip-historical-outcomes"),
    ] = False,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-strategy-evidence",
        func=lambda: build_and_publish_strategy_evidence(
            lake_root=lake_root,
            as_of_date=as_of_date,
            min_live_samples=min_live_samples,
            mode=mode,
            lookback_days=lookback_days,
            include_historical_outcomes=include_historical_outcomes,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-second-stage-alpha-factory")
def build_second_stage_alpha_factory_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
    lookback_days: Annotated[int, typer.Option("--lookback-days", min=1)] = 30,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-second-stage-alpha-factory",
        func=lambda: build_and_publish_second_stage_alpha_factory(
            lake_root=lake_root,
            as_of_date=as_of_date,
            lookback_days=lookback_days,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-alpha-factory")
def build_alpha_factory_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
    lookback_days: Annotated[int, typer.Option("--lookback-days", min=1)] = 30,
    max_candidates: Annotated[int, typer.Option("--max-candidates", min=1, max=200)] = 200,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-alpha-factory",
        func=lambda: build_and_publish_alpha_factory(
            lake_root=lake_root,
            as_of_date=as_of_date,
            lookback_days=lookback_days,
            max_candidates=max_candidates,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-v5-candidate-labels")
def build_v5_candidate_labels_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
    mode: Annotated[str, typer.Option("--mode", help="full or incremental.")] = "incremental",
    lookback_days: Annotated[int, typer.Option("--lookback-days", min=1)] = 8,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-v5-candidate-labels",
        func=lambda: build_and_publish_candidate_labels(
            lake_root=lake_root,
            as_of_date=as_of_date,
            mode=mode,
            lookback_days=lookback_days,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-alpha-discovery-board")
def build_alpha_discovery_board_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
    include_legacy_outcome_counts: Annotated[
        bool,
        typer.Option("--include-legacy-outcome-counts/--skip-legacy-outcome-counts"),
    ] = False,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-alpha-discovery-board",
        func=lambda: build_and_publish_alpha_discovery_board(
            lake_root=lake_root,
            as_of_date=as_of_date,
            include_legacy_outcome_counts=include_legacy_outcome_counts,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-paper-strategy-tracking")
def build_paper_strategy_tracking_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-paper-strategy-tracking",
        func=lambda: build_and_publish_paper_strategy_tracking(
            lake_root=lake_root,
            as_of_date=as_of_date,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-research-portfolio-status")
def build_research_portfolio_status_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-research-portfolio-status",
        func=lambda: build_and_publish_research_portfolio_status(
            lake_root=lake_root,
            as_of_date=as_of_date,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-sol-protect-paper-loss-attribution")
def build_sol_protect_paper_loss_attribution_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-sol-protect-paper-loss-attribution",
        func=lambda: build_and_publish_sol_protect_paper_loss_attribution(
            lake_root=lake_root,
            as_of_date=as_of_date,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-regime-router")
def build_regime_router_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-regime-router",
        func=lambda: build_and_publish_regime_router(
            lake_root=lake_root,
            as_of_date=as_of_date,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-entry-quality")
def build_entry_quality_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
    window_hours: Annotated[int, typer.Option("--window-hours", min=1)] = 24,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-entry-quality",
        func=lambda: build_and_publish_entry_quality(
            lake_root=lake_root,
            as_of_date=as_of_date,
            window_hours=window_hours,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-entry-quality-history")
def build_entry_quality_history_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    start_date: Annotated[str, typer.Option("--start-date", help="UTC start day YYYY-MM-DD")],
    end_date: Annotated[str, typer.Option("--end-date", help="UTC end day YYYY-MM-DD")],
    mode: Annotated[
        str,
        typer.Option("--mode", help="full, recent_7d, recent_30d, or walk_forward"),
    ] = "full",
    cost_mode: Annotated[
        str,
        typer.Option("--cost-mode", help="conservative or quant_lab"),
    ] = "conservative",
    window_hours: Annotated[int, typer.Option("--window-hours", min=1)] = 24,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-entry-quality-history",
        func=lambda: build_and_publish_entry_quality_history(
            lake_root=lake_root,
            start_date=start_date,
            end_date=end_date,
            mode=mode,
            cost_mode=cost_mode,
            window_hours=window_hours,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-btc-probe-exit-policy-review")
def build_btc_probe_exit_policy_review_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
    min_sample_count: Annotated[int, typer.Option("--min-sample-count", min=1)] = 10,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-btc-probe-exit-policy-review",
        func=lambda: build_and_publish_btc_probe_exit_policy_review(
            lake_root=lake_root,
            as_of_date=as_of_date,
            min_sample_count=min_sample_count,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-bnb-swing-exit-policy-review")
def build_bnb_swing_exit_policy_review_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="build-bnb-swing-exit-policy-review",
        func=lambda: build_and_publish_bnb_swing_exit_policy_review(
            lake_root=lake_root,
            as_of_date=as_of_date,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("refresh-research-diagnostics")
def refresh_research_diagnostics_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    as_of_date: Annotated[
        str,
        typer.Option("--date", help="UTC as-of day in YYYY-MM-DD format or auto."),
    ] = "auto",
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="refresh-research-diagnostics",
        func=lambda: refresh_research_diagnostics(
            lake_root=lake_root,
            as_of_date=as_of_date,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("publish-gate-decisions")
def publish_gate_decisions_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    strategy: Annotated[str, typer.Option("--strategy")] = "v5",
) -> None:
    result = publish_gate_decisions_from_evidence(lake_root=lake_root, strategy=strategy)
    typer.echo(result.model_dump_json(indent=2))


@app.command("publish-risk-permission")
def publish_risk_permission_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    strategy: Annotated[str, typer.Option("--strategy")] = "v5",
    version: Annotated[str, typer.Option("--version")] = "5.0.0",
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="publish-risk-permission",
        func=lambda: publish_risk_permission_to_lake(
            lake_root=lake_root,
            strategy=strategy,
            version=version,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("research-health")
def research_health_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    date: Annotated[str | None, typer.Option("--date")] = None,
) -> None:
    result = research_health(lake_root=lake_root, date=date)
    typer.echo(result.model_dump_json(indent=2))


@app.command("enforce-readiness")
def enforce_readiness_command(
    lake_root: Annotated[Path, typer.Option("--lake-root", file_okay=False, dir_okay=True)],
    strategy: Annotated[str, typer.Option("--strategy")] = "v5",
    version: Annotated[str, typer.Option("--version")] = "5.0.0",
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", file_okay=False, dir_okay=True),
    ] = None,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="enforce-readiness",
        func=lambda: write_enforce_readiness_report(
            lake_root=lake_root,
            out_dir=out_dir,
            strategy=strategy,
            version=version,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("run-v5-e2e-contract")
def run_v5_e2e_contract_command(
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="Directory for e2e reports and temporary fixture lake.",
        ),
    ],
    lake_root: Annotated[
        Path | None,
        typer.Option("--lake-root", file_okay=False, dir_okay=True, writable=True),
    ] = None,
) -> None:
    result = run_v5_contract_e2e(out_dir=out_dir, lake_root=lake_root)
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


@app.command("export-daily")
def export_daily(
    export_date: Annotated[str, typer.Option("--date", help="UTC day in YYYY-MM-DD format.")],
    lake_root: Annotated[
        Path,
        typer.Option(
            "--lake-root",
            file_okay=False,
            dir_okay=True,
            help="quant-lab lake root to read from.",
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            file_okay=False,
            dir_okay=True,
            writable=True,
            help="Directory where the expert pack zip will be written.",
        ),
    ],
    profile: Annotated[str, typer.Option("--profile", help="Export profile name.")] = "expert",
    refresh_risk_permission: Annotated[
        bool,
        typer.Option(
            "--refresh-risk-permission/--no-refresh-risk-permission",
            help="Run publish-risk-permission before creating the expert pack.",
        ),
    ] = True,
    risk_strategy: Annotated[str, typer.Option("--risk-strategy")] = "v5",
    risk_version: Annotated[str, typer.Option("--risk-version")] = "5.0.0",
    pre_export_v5_refresh: Annotated[
        bool,
        typer.Option(
            "--pre-export-v5-refresh/--no-pre-export-v5-refresh",
            help="Ingest pending V5 inbox bundles and rebuild V5 research outputs first.",
        ),
    ] = True,
    v5_telemetry_config: Annotated[
        Path | None,
        typer.Option("--v5-telemetry-config", help="Optional V5 telemetry YAML config."),
    ] = None,
    allow_stale_v5: Annotated[
        bool,
        typer.Option(
            "--allow-stale-v5/--no-allow-stale-v5",
            help=(
                "Allow a non-authoritative expert pack when V5 bundle consistency "
                "checks fail."
            ),
        ),
    ] = False,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="export-daily",
        func=lambda: export_daily_pack(
            export_date=export_date,
            lake_root=lake_root,
            out_dir=out_dir,
            profile=profile,
            command_line=sys.argv,
            refresh_risk_permission=refresh_risk_permission,
            risk_strategy=risk_strategy,
            risk_version=risk_version,
            pre_export_v5_refresh=pre_export_v5_refresh,
            v5_telemetry_config=v5_telemetry_config,
            allow_stale_v5=allow_stale_v5,
        ),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("validate-expert-pack")
def validate_expert_pack_command(
    pack_path: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
) -> None:
    result = validate_expert_pack(pack_path)
    typer.echo(result.model_dump_json(indent=2))
    if result.rejected:
        raise typer.Exit(1)


@app.command("pull-v5-bundles")
def pull_v5_bundles(
    config: Annotated[Path, typer.Option("--config", help="V5 telemetry remote YAML config.")],
    remote_host: Annotated[str | None, typer.Option("--remote-host")] = None,
    remote_user: Annotated[str | None, typer.Option("--remote-user")] = None,
    remote_dir: Annotated[Path | None, typer.Option("--remote-dir")] = None,
    local_inbox_dir: Annotated[Path | None, typer.Option("--local-inbox-dir")] = None,
    max_files: Annotated[int | None, typer.Option("--max-files", min=1)] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    cfg = load_v5_telemetry_remote_config(
        config,
        overrides={
            "remote_host": remote_host,
            "remote_user": remote_user,
            "remote_bundle_dir": remote_dir,
            "local_inbox_dir": local_inbox_dir,
            "dry_run": dry_run or None,
        },
    )
    result = RemoteBundlePuller().pull_bundles(cfg, max_files=max_files)
    typer.echo(result.model_dump_json(indent=2))


@app.command("validate-v5-bundle")
def validate_v5_bundle_command(
    bundle_path: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    max_bundle_size_mb: Annotated[int, typer.Option("--max-bundle-size-mb")] = 512,
    max_extracted_size_mb: Annotated[int, typer.Option("--max-extracted-size-mb")] = 2048,
    max_file_count: Annotated[int, typer.Option("--max-file-count")] = 5000,
) -> None:
    limits = BundleLimits(
        max_bundle_size_mb=max_bundle_size_mb,
        max_extracted_size_mb=max_extracted_size_mb,
        max_file_count=max_file_count,
    )
    validation = validate_v5_bundle(bundle_path, limits)
    scan = scan_for_secrets("")
    if validation.valid:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="quant_lab_validate_v5_") as temp_name:
            extract_result = safe_extract_v5_bundle(bundle_path, Path(temp_name), limits)
            scan = scan_for_secrets(Path(extract_result.target_dir))
    payload = {
        "validation": validation.model_dump(mode="json"),
        "secret_scan": scan.model_dump(mode="json"),
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    if validation.rejected or scan.high_severity_count > 0:
        raise typer.Exit(1)


@app.command("ingest-v5-bundle")
def ingest_v5_bundle_command(
    bundle_path: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    lake_root: Annotated[Path, typer.Option("--lake-root")],
    restricted_archive_dir: Annotated[Path, typer.Option("--restricted-archive-dir")],
    redacted_archive_dir: Annotated[Path, typer.Option("--redacted-archive-dir")],
    strategy: Annotated[str, typer.Option("--strategy")] = "v5",
) -> None:
    result = ingest_v5_bundle_file(
        bundle_path=bundle_path,
        lake_root=lake_root,
        restricted_archive_dir=restricted_archive_dir,
        redacted_archive_dir=redacted_archive_dir,
        strategy=strategy,
    )
    typer.echo(result.model_dump_json(indent=2))
    if result.validation.rejected or result.secret_scan.high_severity_count > 0:
        raise typer.Exit(1)


@app.command("ingest-v5-inbox")
def ingest_v5_inbox_command(
    inbox_dir: Annotated[Path, typer.Option("--inbox-dir")],
    lake_root: Annotated[Path, typer.Option("--lake-root")],
    restricted_archive_dir: Annotated[Path, typer.Option("--restricted-archive-dir")],
    redacted_archive_dir: Annotated[Path, typer.Option("--redacted-archive-dir")],
    strategy: Annotated[str, typer.Option("--strategy")] = "v5",
    max_bundles: Annotated[int | None, typer.Option("--max-bundles", min=1)] = None,
    max_scan_bundles: Annotated[int | None, typer.Option("--max-scan-bundles", min=1)] = None,
    newest_first: Annotated[bool, typer.Option("--newest-first/--oldest-first")] = False,
    include_historical_outcomes: Annotated[
        bool,
        typer.Option("--include-historical-outcomes/--skip-historical-outcomes"),
    ] = True,
) -> None:
    result = ingest_v5_inbox_dir(
        inbox_dir=inbox_dir,
        lake_root=lake_root,
        restricted_archive_dir=restricted_archive_dir,
        redacted_archive_dir=redacted_archive_dir,
        strategy=strategy,
        max_bundles=max_bundles,
        max_scan_bundles=max_scan_bundles,
        newest_first=newest_first,
        include_historical_outcomes=include_historical_outcomes,
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("analyze-v5-telemetry")
def analyze_v5_telemetry_command(
    lake_root: Annotated[Path, typer.Option("--lake-root")],
    date: Annotated[str | None, typer.Option("--date")] = None,
    refresh_candidate_gold: Annotated[
        bool,
        typer.Option("--refresh-candidate-gold/--skip-candidate-gold"),
    ] = True,
    compact_output: Annotated[
        bool,
        typer.Option(
            "--compact-output/--full-output",
            help="Emit a one-line operational summary instead of the full analysis payload.",
        ),
    ] = False,
) -> None:
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name="analyze-v5-telemetry",
        func=lambda: analyze_v5_telemetry(
            lake_root=lake_root,
            date=date,
            refresh_candidate_gold=refresh_candidate_gold,
        ),
    )
    if compact_output:
        typer.echo(json.dumps(_compact_v5_telemetry_payload(result), sort_keys=True))
    else:
        typer.echo(result.model_dump_json(indent=2))


def _compact_v5_telemetry_payload(result: object) -> dict[str, object]:
    """Small journald-friendly payload for scheduled telemetry health runs."""

    payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else {}
    keys = [
        "strategy",
        "date",
        "status",
        "latest_bundle_ts",
        "quant_lab_mode",
        "permission_gate_enforced",
        "cost_gate_enforced",
        "unique_request_count",
        "request_success_count",
        "request_error_count",
        "actual_fallback_count",
        "fallback_rate",
        "duplicate_rate",
        "quant_lab_actual_violation_count",
        "quant_lab_hypothetical_violation_count",
        "latest_permission_status",
        "stale_permission_consecutive_count",
    ]
    compact = {key: payload.get(key) for key in keys if key in payload}
    warnings = payload.get("warnings") or []
    critical = payload.get("critical_reasons") or []
    next_actions = payload.get("next_actions") or []
    compact["warning_count"] = len(warnings) if isinstance(warnings, list) else 0
    compact["critical_count"] = len(critical) if isinstance(critical, list) else 0
    compact["next_action_count"] = len(next_actions) if isinstance(next_actions, list) else 0
    compact["warnings"] = warnings[:5] if isinstance(warnings, list) else []
    compact["critical_reasons"] = critical[:5] if isinstance(critical, list) else []
    return compact


@app.command("sync-v5-telemetry")
def sync_v5_telemetry_command(
    config: Annotated[Path, typer.Option("--config", help="V5 telemetry remote YAML config.")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    max_bundles: Annotated[int | None, typer.Option("--max-bundles", min=1)] = None,
    newest_first: Annotated[bool, typer.Option("--newest-first/--oldest-first")] = True,
    max_skipped_files_reported: Annotated[
        int,
        typer.Option("--max-skipped-files-reported", min=0),
    ] = 25,
    remote_max_files: Annotated[int | None, typer.Option("--remote-max-files", min=1)] = None,
    max_scan_bundles: Annotated[int | None, typer.Option("--max-scan-bundles", min=1)] = None,
    include_historical_outcomes: Annotated[
        bool,
        typer.Option("--include-historical-outcomes/--skip-historical-outcomes"),
    ] = False,
    run_analysis_after_sync: Annotated[
        bool,
        typer.Option("--run-analysis-after-sync/--skip-analysis-after-sync"),
    ] = True,
    compact_output: Annotated[
        bool,
        typer.Option(
            "--compact-output/--full-output",
            help="Emit a one-line operational summary instead of nested pull/ingest payloads.",
        ),
    ] = False,
) -> None:
    cfg = load_v5_telemetry_remote_config(
        config,
        overrides={"dry_run": dry_run or None},
    )
    effective_max_bundles = max_bundles
    if effective_max_bundles is None:
        effective_max_bundles = int(os.environ.get("QUANT_LAB_V5_SYNC_MAX_BUNDLES", "1"))
    effective_remote_max_files = remote_max_files
    if effective_remote_max_files is None:
        env_value = os.environ.get("QUANT_LAB_V5_SYNC_REMOTE_MAX_FILES")
        effective_remote_max_files = int(env_value) if env_value else effective_max_bundles
    effective_max_scan_bundles = max_scan_bundles
    if effective_max_scan_bundles is None:
        env_value = os.environ.get("QUANT_LAB_V5_SYNC_MAX_SCAN_BUNDLES")
        effective_max_scan_bundles = int(env_value) if env_value else effective_remote_max_files

    def _run_sync() -> dict[str, object]:
        pull = RemoteBundlePuller().pull_bundles(cfg, max_files=effective_remote_max_files)
        inbox = None
        analysis = None
        if not cfg.dry_run:
            inbox = ingest_v5_inbox_dir(
                inbox_dir=cfg.local_inbox_dir,
                lake_root=cfg.lake_root,
                restricted_archive_dir=cfg.restricted_archive_dir,
                redacted_archive_dir=cfg.redacted_archive_dir,
                strategy=cfg.strategy,
                limits=cfg.bundle_limits,
                max_bundles=effective_max_bundles,
                max_scan_bundles=effective_max_scan_bundles,
                newest_first=newest_first,
                max_skipped_files_reported=max_skipped_files_reported,
                run_analysis=False,
                refresh_candidate_gold=False,
                include_historical_outcomes=include_historical_outcomes,
            )
            if run_analysis_after_sync:
                analysis = analyze_v5_telemetry(
                    lake_root=cfg.lake_root,
                    refresh_candidate_gold=False,
                )
        return {
            "pull": pull.model_dump(mode="json"),
            "inbox": inbox.model_dump(mode="json") if inbox else None,
            "analysis": analysis.model_dump(mode="json") if analysis else None,
            "analysis_after_sync": run_analysis_after_sync,
            "max_bundles": effective_max_bundles,
            "remote_max_files": effective_remote_max_files,
            "max_scan_bundles": effective_max_scan_bundles,
            "newest_first": newest_first,
            "include_historical_outcomes": include_historical_outcomes,
        }

    payload = run_with_job_metrics(
        lake_root=cfg.lake_root,
        job_name="sync-v5-telemetry",
        func=_run_sync,
    )
    output = _compact_v5_sync_payload(payload) if compact_output else payload
    typer.echo(json.dumps(output, indent=None if compact_output else 2, sort_keys=True))


def _compact_v5_sync_payload(payload: dict[str, object]) -> dict[str, object]:
    pull = payload.get("pull") if isinstance(payload.get("pull"), dict) else {}
    inbox = payload.get("inbox") if isinstance(payload.get("inbox"), dict) else {}
    analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
    processed = inbox.get("processed") if isinstance(inbox, dict) else []
    skipped_files = inbox.get("skipped_files") if isinstance(inbox, dict) else []
    pull_warnings = pull.get("warnings") if isinstance(pull, dict) else []
    inbox_warnings = inbox.get("warnings") if isinstance(inbox, dict) else []
    processed_bundles: list[str] = []
    if isinstance(processed, list):
        for item in processed[:5]:
            if isinstance(item, dict) and item.get("bundle_name"):
                processed_bundles.append(str(item["bundle_name"]))
    warnings: list[object] = []
    if isinstance(pull_warnings, list):
        warnings.extend(pull_warnings[:5])
    if isinstance(inbox_warnings, list):
        warnings.extend(inbox_warnings[:5])
    operational_warnings = [
        warning
        for warning in warnings
        if not _v5_sync_warning_is_expected_limit_notice(warning)
    ]
    return {
        "analysis_after_sync": payload.get("analysis_after_sync"),
        "include_historical_outcomes": payload.get("include_historical_outcomes"),
        "max_bundles": payload.get("max_bundles"),
        "remote_max_files": payload.get("remote_max_files"),
        "max_scan_bundles": payload.get("max_scan_bundles"),
        "pulled_count": len(pull.get("pulled_files") or []) if isinstance(pull, dict) else 0,
        "skipped_pull_count": len(pull.get("skipped_files") or []) if isinstance(pull, dict) else 0,
        "processed_count": len(processed) if isinstance(processed, list) else 0,
        "processed_bundles": processed_bundles,
        "skipped_inbox_count": len(skipped_files) if isinstance(skipped_files, list) else 0,
        "pull_warning_count": len(pull_warnings) if isinstance(pull_warnings, list) else 0,
        "inbox_warning_count": len(inbox_warnings) if isinstance(inbox_warnings, list) else 0,
        "scan_limited": any(
            _v5_sync_warning_is_expected_limit_notice(warning) for warning in warnings
        ),
        "warnings": operational_warnings,
        "analysis_status": analysis.get("status") if isinstance(analysis, dict) else None,
        "latest_bundle_ts": (
            analysis.get("latest_bundle_ts") if isinstance(analysis, dict) else None
        ),
    }


def _v5_sync_warning_is_expected_limit_notice(value: object) -> bool:
    rendered = str(value)
    return rendered.startswith(
        (
            "max_scan_bundles_limit_applied:",
            "max_bundles_limit_applied:",
        )
    )


def main() -> None:
    app()


def _compact_dataset_target(lake_root: Path, dataset: str) -> tuple[Path, list[str]]:
    normalized = dataset.strip().replace("\\", "/")
    targets = {
        "okx_public_ws": (
            lake_root / "bronze" / "okx_public_ws",
            ["day", "channel", "inst_id"],
        ),
        "bronze/okx_public_ws": (
            lake_root / "bronze" / "okx_public_ws",
            ["day", "channel", "inst_id"],
        ),
        "trade_print": (lake_root / "silver" / "trade_print", ["day", "symbol"]),
        "silver/trade_print": (lake_root / "silver" / "trade_print", ["day", "symbol"]),
        "orderbook_snapshot": (
            lake_root / "silver" / "orderbook_snapshot",
            ["day", "symbol", "channel"],
        ),
        "silver/orderbook_snapshot": (
            lake_root / "silver" / "orderbook_snapshot",
            ["day", "symbol", "channel"],
        ),
    }
    if normalized in targets:
        return targets[normalized]
    return lake_root / normalized, []


if __name__ == "__main__":
    main()
