import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from quant_lab.contracts.models import AlphaEvidence, AlphaResearchSpec
from quant_lab.costs.calibrate import calibrate_costs_for_day
from quant_lab.costs.health import read_cost_health_daily
from quant_lab.data.lake import compact_parquet_dataset
from quant_lab.e2e import run_v5_contract_e2e
from quant_lab.export.daily import export_daily_pack, validate_expert_pack
from quant_lab.features.publish import feature_health
from quant_lab.features.publish import publish_features as publish_feature_values
from quant_lab.gates.defaults import evaluate_alpha_gate
from quant_lab.ingest.okx_public import (
    MARKET_BAR_DATASET,
    OKXPublicClient,
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
from quant_lab.ops.lake_health import write_lake_file_health_daily
from quant_lab.ops.metrics import api_metrics_summary, job_run_summary, run_with_job_metrics
from quant_lab.reports.enforce_readiness import write_enforce_readiness_report
from quant_lab.research.alpha_discovery import build_and_publish_alpha_discovery_board
from quant_lab.research.bootstrap_gold import bootstrap_gold_health
from quant_lab.research.candidate_labels import build_and_publish_candidate_labels
from quant_lab.research.publish import (
    build_and_publish_alpha_evidence,
    publish_gate_decisions_from_evidence,
    research_health,
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
    decision = evaluate_alpha_gate(AlphaEvidence.example_live_ready())
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
    ] = 10.0,
    flush_max_messages: Annotated[
        int,
        typer.Option("--flush-max-messages", min=1),
    ] = 100,
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
) -> None:
    result = write_lake_file_health_daily(lake_root)
    typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))


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
) -> None:
    dataset_path, partition_by = _compact_dataset_target(lake_root, dataset)
    result = run_with_job_metrics(
        lake_root=lake_root,
        job_name=f"compact-lake-dataset:{dataset}",
        func=lambda: compact_parquet_dataset(
            dataset_path,
            partition_by=partition_by,
            target_rows_per_file=target_rows_per_file,
            max_source_files_per_batch=max_source_files_per_batch,
        ),
    )
    typer.echo(json.dumps(result.__dict__, indent=2, sort_keys=True))


@app.command("ops-summary")
def ops_summary_command(
    lake_root: Annotated[
        Path,
        typer.Option("--lake-root", file_okay=False, dir_okay=True),
    ],
    day: Annotated[str | None, typer.Option("--day")] = None,
) -> None:
    payload = {
        "api_metrics": api_metrics_summary(lake_root, day=day),
        "job_runs": job_run_summary(lake_root, day=day),
        "lake_file_health": write_lake_file_health_daily(lake_root),
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


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
    result = RemoteBundlePuller().pull_bundles(cfg)
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
    typer.echo(result.model_dump_json(indent=2))


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
    include_historical_outcomes: Annotated[
        bool,
        typer.Option("--include-historical-outcomes/--skip-historical-outcomes"),
    ] = False,
) -> None:
    cfg = load_v5_telemetry_remote_config(
        config,
        overrides={"dry_run": dry_run or None},
    )
    effective_max_bundles = max_bundles
    if effective_max_bundles is None:
        effective_max_bundles = int(os.environ.get("QUANT_LAB_V5_SYNC_MAX_BUNDLES", "1"))
    def _run_sync() -> dict[str, object]:
        pull = RemoteBundlePuller().pull_bundles(cfg)
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
                newest_first=newest_first,
                max_skipped_files_reported=max_skipped_files_reported,
                run_analysis=False,
                refresh_candidate_gold=False,
                include_historical_outcomes=include_historical_outcomes,
            )
            analysis = analyze_v5_telemetry(
                lake_root=cfg.lake_root,
                refresh_candidate_gold=False,
            )
        return {
            "pull": pull.model_dump(mode="json"),
            "inbox": inbox.model_dump(mode="json") if inbox else None,
            "analysis": analysis.model_dump(mode="json") if analysis else None,
            "max_bundles": effective_max_bundles,
            "newest_first": newest_first,
            "include_historical_outcomes": include_historical_outcomes,
        }

    payload = run_with_job_metrics(
        lake_root=cfg.lake_root,
        job_name="sync-v5-telemetry",
        func=_run_sync,
    )
    typer.echo(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
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
