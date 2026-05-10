from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from quant_lab.web.pages import (
    alpha_gates,
    cost_model,
    data_health,
    expert_exports,
    market_regime,
    okx_collectors,
    overview,
    strategy_consumers,
    v5_telemetry,
)

PageRenderer = Callable[[str | Path, Any | None], None]

PAGES: dict[str, PageRenderer] = {
    "Overview": overview.render,
    "Data Health": data_health.render,
    "OKX Collectors": okx_collectors.render,
    "Market Regime": market_regime.render,
    "Cost Model": cost_model.render,
    "Alpha Gates": alpha_gates.render,
    "Strategy Consumers": strategy_consumers.render,
    "V5 Telemetry": v5_telemetry.render,
    "Expert Exports": expert_exports.render,
}


def render_dashboard(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = _streamlit(st_module)
    st.set_page_config(page_title="quant-lab", layout="wide")
    st.sidebar.title("quant-lab")
    st.sidebar.caption("OKX-first read-only research observability.")
    selected_page = st.sidebar.selectbox("Page", list(PAGES))
    selected_lake_root = st.sidebar.text_input("Lake root", value=str(lake_root))
    st.sidebar.caption("No strategy or exchange state mutation is available in this dashboard.")
    PAGES[selected_page](Path(selected_lake_root), st)


def streamlit_entry(argv: Sequence[str] | None = None) -> None:
    args = _parse_streamlit_args(argv)
    render_dashboard(args.lake_root)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_launcher_args(argv)
    app_path = Path(__file__).resolve()
    env = os.environ.copy()
    env["QUANT_LAB_LAKE_ROOT"] = str(args.lake_root)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--",
        "--lake-root",
        str(args.lake_root),
    ]
    return subprocess.run(command, env=env, check=False).returncode


def _parse_launcher_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="qlab-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument(
        "--lake-root",
        default=os.environ.get("QUANT_LAB_LAKE_ROOT", "/var/lib/quant-lab/lake"),
    )
    return parser.parse_args(argv)


def _parse_streamlit_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lake-root",
        default=os.environ.get("QUANT_LAB_LAKE_ROOT", "/var/lib/quant-lab/lake"),
    )
    return parser.parse_args(argv)


def _streamlit(st_module: Any | None = None) -> Any:
    if st_module is not None:
        return st_module
    import streamlit as st

    return st


if __name__ == "__main__":
    streamlit_entry()
