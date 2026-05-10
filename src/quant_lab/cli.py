import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from quant_lab.contracts.models import AlphaEvidence
from quant_lab.costs.calibrate import calibrate_costs_for_day
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
)
from quant_lab.ingest.okx_ws_public import collect_okx_public_ws
from quant_lab.ingest.v5_reports import inspect_v5_reports, publish_v5_reports_to_lake

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
) -> None:
    result = calibrate_costs_for_day(lake_root=lake_root, day=day)
    typer.echo(result.model_dump_json(indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
