from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.web import readers
from quant_lab.web.pages._common import (
    display_unknown,
    display_value,
    lake_caption,
    show_frame,
    show_warnings,
    streamlit_module,
)


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    snapshot = readers.dashboard_overview(lake_root)

    st.title("quant-lab 概览")
    lake_caption(st, lake_root)
    st.caption("OKX 优先研究数据和策略权限的只读观测面板。")

    diagnostics = snapshot["diagnostics"]
    st.subheader("诊断")
    st.metric("Lake 根目录", diagnostics["lake_root"])
    st.metric("Lake 根目录存在", display_value(diagnostics["lake_root_exists"]))
    st.metric("Lake Parquet 文件数", diagnostics["parquet_file_count"])
    st.metric(
        "最新 market_bar 时间",
        display_unknown(diagnostics["latest_market_bar_ts"]),
    )
    show_frame(
        st,
        diagnostics["datasets"],
        "暂无核心数据集诊断信息。",
    )
    st.metric("最新 V5 bundle", display_unknown(diagnostics.get("latest_v5_bundle_ts")))
    st.metric("最新专家包", display_unknown(diagnostics.get("latest_expert_pack")))
    show_frame(
        st,
        diagnostics.get("suggested_commands", pl.DataFrame()),
        "暂无建议命令。",
    )
    show_warnings(st, diagnostics["warnings"])

    st.metric("quant-lab 状态", display_value(snapshot["status"]))
    st.metric("V5 权限", display_value(snapshot["v5_permission"]))
    st.metric("V7 权限", display_value(snapshot["v7_permission"]))
    st.metric("最新 market_bar", display_unknown(snapshot["latest_market_bar_ts"]))
    st.metric("缺失 K 线比例", f"{snapshot['missing_bar_ratio']:.2%}")
    fallback_ratio = snapshot["cost_fallback_ratio"]
    st.metric("成本回退比例", "不适用" if fallback_ratio is None else f"{fallback_ratio:.2%}")

    collector_rows = [
        {"collector": "OKX 公共 REST", "status": display_value(snapshot["okx_public_rest_status"])},
        {
            "collector": "OKX 公共 WebSocket",
            "status": display_value(snapshot["okx_public_ws_status"]),
        },
        {"collector": "OKX 只读私有接口", "status": display_value(snapshot["okx_readonly_status"])},
    ]
    st.subheader("采集器状态")
    show_frame(st, pl.DataFrame(collector_rows), "暂无采集器状态。")

    st.subheader("Alpha 门控数量")
    gate_counts = pl.DataFrame(
        [
            {"status": status, "count": count}
            for status, count in snapshot["alpha_gate_counts"].items()
        ]
    )
    show_frame(st, gate_counts, "暂无 gate_decision 数据。")

    st.subheader("最新专家包")
    st.write(snapshot["latest_expert_pack"] or "未找到专家包。")
    diagnostic_warnings = set(diagnostics["warnings"])
    show_warnings(
        st,
        [warning for warning in snapshot["warnings"] if warning not in diagnostic_warnings],
    )
