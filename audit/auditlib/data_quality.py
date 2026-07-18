# ruff: noqa: E501
"""Data-quality evidence for market, funding, and production execution data."""

from __future__ import annotations

import html
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from .paths import ARTIFACTS, CANDLES_DIR, FUNDING_DIR, REPORTS, snapshot_dir

BAR_SECONDS = 3600
EXTREME_RET_THRESHOLD = 0.25
GAP_BARS_THRESHOLD = 48
SEVERITY_ORDER = {"ok": 0, "warning": 1, "critical": 2}


def _max_severity(left: str, right: str) -> str:
    return max((left, right), key=SEVERITY_ORDER.__getitem__)


def _load_parquet_tree(path: Path) -> pl.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.scan_parquet([str(file) for file in files]).collect()


def _epoch_ms(series: pl.Series) -> np.ndarray:
    if series.dtype in (pl.Datetime, pl.Date):
        return series.cast(pl.Datetime("ms")).cast(pl.Int64).to_numpy()
    return series.cast(pl.Int64, strict=False).to_numpy()


def _iso_from_ms(value: int) -> str:
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC).isoformat()


def check_candle_frame(df: pl.DataFrame, inst_id: str) -> dict:
    """Check one symbol while preserving the source order for order diagnostics."""
    if df.is_empty():
        return {
            "source": "okx_history_candles",
            "symbol": inst_id,
            "status": "empty",
            "severity": "critical",
        }

    original_ts_ms = _epoch_ms(df["ts"])
    out_of_order = int((np.diff(original_ts_ms) < 0).sum())
    frame = df.with_columns(pl.Series("_ts_ms", original_ts_ms)).sort("_ts_ms")
    ts_ms = frame["_ts_ms"].to_numpy()
    first_ms, last_ms = int(ts_ms[0]), int(ts_ms[-1])
    actual = frame.height
    unique = frame["_ts_ms"].n_unique()
    theoretical = (last_ms - first_ms) // (BAR_SECONDS * 1000) + 1
    missing_rate = max(0.0, 1.0 - unique / theoretical) if theoretical > 0 else 0.0
    duplicate_rate = 1.0 - unique / actual if actual else 0.0

    nonpositive_price = frame.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
    ).height
    ohlc_error = frame.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
    ).height
    volume_col = "quote_volume" if "quote_volume" in frame.columns else "volume"
    zero_volume = frame.filter(pl.col(volume_col).fill_null(0) <= 0).height
    volume_values = frame[volume_col].fill_null(0).to_numpy()
    zero_run = max_zero_run = leading_zero_run = 0
    still_leading = True
    for value in volume_values:
        zero_run = zero_run + 1 if value <= 0 else 0
        max_zero_run = max(max_zero_run, zero_run)
        if still_leading and value <= 0:
            leading_zero_run += 1
        else:
            still_leading = False

    returns = frame.select((pl.col("close") / pl.col("close").shift(1) - 1.0).alias("r"))[
        "r"
    ].to_numpy()
    extreme_returns = (
        int(np.sum(np.abs(returns[1:]) > EXTREME_RET_THRESHOLD)) if len(returns) > 1 else 0
    )
    gaps = np.diff(ts_ms) // (BAR_SECONDS * 1000)
    max_gap = int(gaps.max()) if gaps.size else 0
    long_gaps = int(np.sum(gaps > GAP_BARS_THRESHOLD)) if gaps.size else 0
    unclosed = 0
    if "confirm" in frame.columns:
        unclosed = frame.filter(pl.col("confirm").cast(pl.Utf8) != "1").height

    issues: list[str] = []
    severity = "ok"
    if missing_rate > 0.05:
        issues.append(f"missing_rate={missing_rate:.3f}")
        severity = _max_severity(severity, "critical" if missing_rate > 0.30 else "warning")
    if duplicate_rate > 0.001:
        issues.append(f"duplicate_rate={duplicate_rate:.4f}")
        severity = "critical"
    if out_of_order:
        issues.append(f"out_of_order={out_of_order}")
        severity = "critical"
    if nonpositive_price:
        issues.append(f"nonpositive_price={nonpositive_price}")
        severity = "critical"
    if ohlc_error:
        issues.append(f"ohlc_logic_error={ohlc_error}")
        severity = _max_severity(severity, "warning")
    if unclosed:
        issues.append(f"unclosed_bars={unclosed}")
        severity = _max_severity(severity, "warning")
    if max_gap > GAP_BARS_THRESHOLD:
        issues.append(f"max_gap_bars={max_gap}")
        severity = _max_severity(severity, "warning")
    if extreme_returns:
        issues.append(f"extreme_returns={extreme_returns}")
    if leading_zero_run >= 24:
        issues.append(f"possible_prelisting_pseudo_bars={leading_zero_run}")
        severity = _max_severity(severity, "warning")

    return {
        "source": "okx_history_candles",
        "symbol": inst_id,
        "status": "ok",
        "severity": severity,
        "first_ts": _iso_from_ms(first_ms),
        "last_ts": _iso_from_ms(last_ms),
        "theoretical_bars": int(theoretical),
        "actual_bars": actual,
        "unique_bars": int(unique),
        "missing_rate": round(missing_rate, 6),
        "duplicate_rate": round(duplicate_rate, 6),
        "out_of_order": out_of_order,
        "nonpositive_prices": nonpositive_price,
        "ohlc_logic_errors": ohlc_error,
        "zero_volume_bars": zero_volume,
        "max_zero_volume_run": max_zero_run,
        "possible_prelisting_pseudo_bars": leading_zero_run,
        "extreme_return_bars": extreme_returns,
        "max_gap_bars": max_gap,
        "long_gaps_gt48h": long_gaps,
        "unclosed_bars": unclosed,
        "issues": ";".join(issues),
    }


def check_symbol_candles(inst_dir: Path) -> dict:
    return check_candle_frame(_load_parquet_tree(inst_dir), inst_dir.name.replace("inst_id=", ""))


def check_funding_frame(
    df: pl.DataFrame, inst_id: str, *, required_history_days: int = 730
) -> dict:
    if df.is_empty():
        return {
            "source": "okx_funding_history",
            "symbol": inst_id,
            "funding_rows": 0,
            "severity": "critical",
            "issues": "no_funding_data",
        }
    time_col = "funding_time" if "funding_time" in df.columns else "funding_ts"
    original = _epoch_ms(df[time_col])
    out_of_order = int((np.diff(original) < 0).sum())
    frame = df.with_columns(pl.Series("_ts_ms", original)).sort("_ts_ms")
    ts = frame["_ts_ms"].to_numpy()
    unique_ts = np.unique(ts)
    first_ms, last_ms = int(unique_ts[0]), int(unique_ts[-1])
    span_days = (last_ms - first_ms) / 86_400_000
    duplicate_rate = 1.0 - len(unique_ts) / frame.height
    gaps_hours = np.diff(unique_ts) / 3_600_000
    positive_gaps = gaps_hours[gaps_hours > 0]
    cadence_hours = float(np.median(positive_gaps)) if positive_gaps.size else 0.0
    expected = (
        ((last_ms - first_ms) / (cadence_hours * 3_600_000) + 1)
        if cadence_hours > 0
        else frame.height
    )
    coverage = min(1.0, len(unique_ts) / expected) if expected else 0.0

    issues: list[str] = []
    severity = "ok"
    if span_days < required_history_days:
        issues.append(f"short_history_days={span_days:.1f}")
        severity = "critical"
    if duplicate_rate > 0.001:
        issues.append(f"duplicate_rate={duplicate_rate:.4f}")
        severity = "critical"
    if out_of_order:
        issues.append(f"out_of_order={out_of_order}")
        severity = "critical"
    if coverage < 0.90:
        issues.append(f"cadence_coverage={coverage:.2f}")
        severity = _max_severity(severity, "warning")
    if cadence_hours <= 0 or cadence_hours > 24:
        issues.append(f"irregular_cadence_hours={cadence_hours:.2f}")
        severity = _max_severity(severity, "warning")

    return {
        "source": "okx_funding_history",
        "symbol": inst_id,
        "funding_rows": frame.height,
        "first_ts": _iso_from_ms(first_ms),
        "last_ts": _iso_from_ms(last_ms),
        "span_days": round(span_days, 1),
        "inferred_cadence_hours": round(cadence_hours, 3),
        "cadence_coverage": round(coverage, 4),
        "duplicate_rate": round(duplicate_rate, 6),
        "out_of_order": out_of_order,
        "severity": severity,
        "issues": ";".join(issues),
    }


def check_funding(funding_dir: Path) -> dict:
    return check_funding_frame(
        _load_parquet_tree(funding_dir), funding_dir.name.replace("inst_id=", "")
    )


_EXECUTION_TABLES = {
    "fills.sqlite": {
        "table": "fills",
        "timestamp": "ts_ms",
        "required": ["inst_id", "trade_id", "ts_ms", "side", "fill_px", "fill_sz", "fee"],
        "positive": ["fill_px", "fill_sz"],
    },
    "orders.sqlite": {
        "table": "orders",
        "timestamp": "created_ts",
        "required": ["cl_ord_id", "inst_id", "side", "state", "created_ts"],
        "positive": [],
    },
    "bills.sqlite": {
        "table": "bills",
        "timestamp": "ts_ms",
        "required": ["bill_id", "ts_ms", "ccy", "bal_chg", "type", "source"],
        "positive": [],
    },
}


def check_execution_databases(db_dir: Path) -> pl.DataFrame:
    rows: list[dict] = []
    for filename, spec in _EXECUTION_TABLES.items():
        path = db_dir / filename
        if not path.is_file():
            rows.append(
                {
                    "source": filename,
                    "table": spec["table"],
                    "rows": 0,
                    "severity": "critical",
                    "issues": "database_missing",
                }
            )
            continue
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            table = spec["table"].replace('"', '""')
            columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
            missing_columns = sorted(set(spec["required"]) - columns)
            count = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            null_conditions = [
                f'"{column}" IS NULL' for column in spec["required"] if column in columns
            ]
            null_rows = (
                int(
                    connection.execute(
                        f'SELECT COUNT(*) FROM "{table}" WHERE ' + " OR ".join(null_conditions)
                    ).fetchone()[0]
                )
                if null_conditions
                else 0
            )
            nonpositive_conditions = [
                f'CAST("{column}" AS REAL) <= 0' for column in spec["positive"] if column in columns
            ]
            nonpositive_rows = (
                int(
                    connection.execute(
                        f'SELECT COUNT(*) FROM "{table}" WHERE '
                        + " OR ".join(nonpositive_conditions)
                    ).fetchone()[0]
                )
                if nonpositive_conditions
                else 0
            )
            first_ts = last_ts = None
            timestamp_col = spec["timestamp"]
            if timestamp_col in columns:
                first_ts, last_ts = connection.execute(
                    f'SELECT MIN("{timestamp_col}"), MAX("{timestamp_col}") FROM "{table}"'
                ).fetchone()
            issues = []
            severity = "ok"
            if count == 0:
                issues.append("empty_table")
                severity = "critical"
            if missing_columns:
                issues.append("missing_columns=" + ",".join(missing_columns))
                severity = "critical"
            if null_rows:
                issues.append(f"required_field_null_rows={null_rows}")
                severity = "critical"
            if nonpositive_rows:
                issues.append(f"nonpositive_execution_rows={nonpositive_rows}")
                severity = "critical"
            rows.append(
                {
                    "source": filename,
                    "table": spec["table"],
                    "rows": count,
                    "first_timestamp_raw": first_ts,
                    "last_timestamp_raw": last_ts,
                    "missing_required_columns": ",".join(missing_columns),
                    "required_field_null_rows": null_rows,
                    "nonpositive_execution_rows": nonpositive_rows,
                    "severity": severity,
                    "issues": ";".join(issues),
                }
            )
        finally:
            connection.close()
    return pl.DataFrame(rows)


def _html_table(frame: pl.DataFrame, limit: int = 200) -> str:
    if frame.is_empty():
        return "<p>No rows.</p>"
    columns = frame.columns
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in frame.head(limit).iter_rows(named=True):
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def run_data_quality() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    candle_dirs = sorted(path for path in CANDLES_DIR.iterdir() if path.is_dir())
    candles = pl.DataFrame([check_symbol_candles(path) for path in candle_dirs])
    funding_dirs = sorted(path for path in FUNDING_DIR.iterdir() if path.is_dir())
    funding_rows = [check_funding(path) for path in funding_dirs]
    funding = pl.DataFrame(funding_rows) if funding_rows else pl.DataFrame()
    execution = check_execution_databases(snapshot_dir() / "v5" / "db")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    candles.write_csv(ARTIFACTS / "data_quality_summary.csv")
    if not funding.is_empty():
        funding.write_csv(ARTIFACTS / "funding_quality_summary.csv")
    execution.write_csv(ARTIFACTS / "execution_data_quality_summary.csv")

    latest_first = candles.filter(pl.col("status") == "ok")["first_ts"].max()
    earliest_last = candles.filter(pl.col("status") == "ok")["last_ts"].min()
    candle_counts = candles.group_by("severity").len().sort("severity").to_dicts()
    funding_counts = (
        funding.group_by("severity").len().sort("severity").to_dicts()
        if not funding.is_empty()
        else []
    )
    execution_counts = execution.group_by("severity").len().sort("severity").to_dicts()
    lines = [
        "# Data Quality Report",
        "",
        f"- generated_at: {datetime.now(UTC).isoformat()}",
        f"- candle symbols: {candles.height}; severity counts: {candle_counts}",
        f"- funding symbols: {funding.height}; severity counts: {funding_counts}",
        f"- production execution sources: {execution.height}; severity counts: {execution_counts}",
        f"- common alignment window: latest first={latest_first}; earliest last={earliest_last}",
        "- candle ranking and features exclude the decision day; funding cadence is inferred per symbol.",
        "- funding history shorter than 730 days is critical and propagates to funding_fade=INCONCLUSIVE.",
        "- candidate instruments were seeded from currently listed OKX products; delisted-before-snapshot assets are not reconstructable (survivorship limitation).",
        "",
        "## Candle warnings and critical rows",
        "",
    ]
    bad = candles.filter(pl.col("severity") != "ok").sort(["severity", "symbol"])
    lines.extend(
        [
            f"- {row['symbol']}: {row['severity']} {row['issues']}"
            for row in bad.iter_rows(named=True)
        ]
        or ["- none"]
    )
    lines.extend(["", "## Funding coverage", ""])
    lines.extend(
        [
            f"- {row['symbol']}: {row['span_days']}d, cadence={row['inferred_cadence_hours']}h, "
            f"coverage={row['cadence_coverage']}, {row['severity']} {row['issues']}"
            for row in funding.sort("span_days").iter_rows(named=True)
        ]
        or ["- no funding data"]
    )
    lines.extend(["", "## Execution record completeness", ""])
    lines.extend(
        [
            f"- {row['source']}/{row['table']}: rows={row['rows']}, {row['severity']} {row['issues']}"
            for row in execution.iter_rows(named=True)
        ]
    )
    (REPORTS / "data_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    dashboard = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Data Quality</title>
<style>body{{font-family:system-ui;background:#0b1220;color:#e5edf7;margin:24px}}table{{border-collapse:collapse;font-size:12px;width:100%;margin-bottom:30px}}th,td{{border:1px solid #334155;padding:6px;text-align:left}}th{{position:sticky;top:0;background:#1e293b}}h1,h2{{color:#7dd3fc}}.note{{padding:12px;background:#172033;border-left:4px solid #f59e0b}}</style></head><body>
<h1>Alpha Audit — Data Quality</h1><p class=\"note\">Funding below 730 days and historical-universe survivorship limits propagate into final factor decisions.</p>
<h2>Candles ({candles.height})</h2>{_html_table(candles)}
<h2>Funding ({funding.height})</h2>{_html_table(funding)}
<h2>Execution records</h2>{_html_table(execution)}</body></html>"""
    (REPORTS / "data_quality_dashboard.html").write_text(dashboard, encoding="utf-8")
    return candles, funding, execution
