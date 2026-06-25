from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from quant_lab.web.pages import (
    alpha_gates,
    bigscreen_v2,
    cost_model,
    data_health,
    expert_exports,
    market_regime,
    okx_collectors,
    overview,
    strategy_consumers,
    v5_telemetry,
    web_performance,
)

PageRenderer = Callable[[str | Path, Any | None], None]

PAGES: dict[str, PageRenderer] = {
    "🚀 V2 大屏": bigscreen_v2.render,
    "📌 概览": overview.render,
    "🧭 策略机会": alpha_gates.render,
    "📦 专家包导出": expert_exports.render,
    "📡 V5 遥测": v5_telemetry.render,
    "💰 成本模型": cost_model.render,
    "📈 市场状态": market_regime.render,
    "🔌 OKX 采集器": okx_collectors.render,
    "🩺 数据健康": data_health.render,
    "🧪 Web 性能": web_performance.render,
    "🔐 策略消费者": strategy_consumers.render,
}


def render_dashboard(lake_root: str | Path, st_module: Any | None = None) -> None:
    from quant_lab.web import perf

    st = _streamlit(st_module)
    st.set_page_config(page_title="quant-lab", layout="wide")
    st.sidebar.title("quant-lab")
    st.sidebar.caption("OKX 优先的只读研究观测台")
    selected_page = st.sidebar.radio("页面", list(PAGES))
    selected_lake_root = st.sidebar.text_input("Lake 根目录", value=str(lake_root))
    st.sidebar.caption("只读：不下单、不撤单、不修改策略或交易所状态。")
    page_key = _resolve_page_key(selected_page)
    start = perf_counter()
    PAGES[page_key](Path(selected_lake_root), st)
    perf.record_event(
        "page_render",
        page_name=page_key,
        elapsed_ms=(perf_counter() - start) * 1000,
    )


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
    return _run_streamlit_process(command, env=env)


def _run_streamlit_process(command: Sequence[str], *, env: dict[str, str]) -> int:
    process = subprocess.Popen(command, env=env)
    received_signal: int | None = None
    stop_started_at: float | None = None
    stop_timeout_seconds = 10.0

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal received_signal, stop_started_at
        received_signal = signum
        if stop_started_at is None:
            stop_started_at = perf_counter()
            try:
                process.terminate()
            except ProcessLookupError:
                return

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        while True:
            return_code = process.poll()
            if return_code is not None:
                return 0 if received_signal is not None else int(return_code)
            if (
                stop_started_at is not None
                and perf_counter() - stop_started_at >= stop_timeout_seconds
            ):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            sleep(0.2)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


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
