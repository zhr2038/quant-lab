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
    diagnostics = snapshot["diagnostics"]

    st.title("📌 quant-lab 概览")
    lake_caption(st, lake_root)
    st.caption("OKX 优先、只读研究中台。重点看数据新鲜度、V5 遥测、成本质量和策略机会。")

    st.subheader("🚦 关键状态")
    st.metric("中台状态", display_value(snapshot["status"]))
    st.metric("V5 风险权限", display_value(snapshot["v5_permission"]))
    st.metric("最新行情 K 线", display_unknown(snapshot["latest_market_bar_ts"]))
    st.metric("最新 V5 数据包", display_unknown(diagnostics.get("latest_v5_bundle_ts")))
    fallback_ratio = snapshot["cost_fallback_ratio"]
    st.metric("成本回退比例", "不适用" if fallback_ratio is None else f"{fallback_ratio:.2%}")
    st.metric("Lake Parquet 文件数", diagnostics["parquet_file_count"])

    st.subheader("🎯 策略机会与入场质量")
    show_frame(
        st,
        snapshot.get("strategy_opportunity_advisory", pl.DataFrame()),
        "暂无策略机会建议。请先运行 alpha discovery / entry quality 构建任务。",
    )
    show_frame(
        st,
        snapshot.get("entry_quality_advisory", pl.DataFrame()),
        "暂无入场质量建议。",
    )

    st.subheader("📡 采集器状态")
    collector_rows = [
        {"collector": "OKX 公共 REST", "status": display_value(snapshot["okx_public_rest_status"])},
        {
            "collector": "OKX 公共 WebSocket",
            "status": display_value(snapshot["okx_public_ws_status"]),
        },
        {"collector": "OKX 只读私有接口", "status": display_value(snapshot["okx_readonly_status"])},
    ]
    show_frame(st, pl.DataFrame(collector_rows), "暂无采集器状态。")

    st.subheader("🧭 因子 / 候选状态")
    gate_counts = pl.DataFrame(
        [
            {"status": status, "count": count}
            for status, count in snapshot["alpha_gate_counts"].items()
        ]
    )
    show_frame(st, gate_counts, "暂无门控决策数据。")

    stale_or_missing = _focus_diagnostics(diagnostics["datasets"])
    if not stale_or_missing.is_empty():
        st.subheader("🩺 需要处理的数据集")
        show_frame(st, stale_or_missing, "核心数据集当前正常。")

    suggested = diagnostics.get("suggested_commands", pl.DataFrame())
    if not suggested.is_empty():
        st.subheader("🛠 建议命令")
        show_frame(st, suggested, "暂无建议命令。")

    st.subheader("📦 最新专家包")
    st.write(snapshot["latest_expert_pack"] or "未找到专家包。")

    show_warnings(st, snapshot["warnings"])


def _focus_diagnostics(datasets: pl.DataFrame) -> pl.DataFrame:
    if datasets.is_empty() or "freshness_status" not in datasets.columns:
        return datasets
    focus_statuses = {"missing", "stale", "delayed", "unknown"}
    return datasets.filter(pl.col("freshness_status").is_in(focus_statuses))
