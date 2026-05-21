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
    "📌 概览": overview.render,
    "🧭 策略机会": alpha_gates.render,
    "📦 专家包导出": expert_exports.render,
    "📡 V5 遥测": v5_telemetry.render,
    "💰 成本模型": cost_model.render,
    "📈 市场状态": market_regime.render,
    "🔌 OKX 采集器": okx_collectors.render,
    "🩺 数据健康": data_health.render,
    "🔐 策略消费者": strategy_consumers.render,
}


def render_dashboard(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = _streamlit(st_module)
    st.set_page_config(page_title="quant-lab", layout="wide")
    st.sidebar.title("quant-lab")
    st.sidebar.caption("OKX 优先的只读研究观测台")
    selected_page = st.sidebar.radio("页面", list(PAGES))
    selected_lake_root = st.sidebar.text_input("Lake 根目录", value=str(lake_root))
    st.sidebar.caption("只读：不下单、不撤单、不修改策略或交易所状态。")
    page_key = _resolve_page_key(selected_page)
    PAGES[page_key](Path(selected_lake_root), st)


def _resolve_page_key(selected_page: str) -> str:
    if selected_page in PAGES:
        return selected_page
    for key in PAGES:
        if key.endswith(selected_page):
            return key
    return next(iter(PAGES))


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
        "--client.showSidebarNavigation",
        "false",
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
