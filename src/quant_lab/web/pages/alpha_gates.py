from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import (
    display_value,
    lake_caption,
    show_frame,
    show_warnings,
    streamlit_module,
)


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.alpha_gate_summary(lake_root)

    st.title("🧭 策略机会")
    lake_caption(st, lake_root)
    st.caption("优先展示可给 V5 读取的策略候选建议；底层样本表不在首页铺开。")

    st.subheader("🎯 V5 可读策略机会")
    show_frame(
        st,
        summary["strategy_opportunity_advisory"],
        "暂无策略机会建议数据。",
    )

    st.subheader("📊 候选决策面板")
    for decision, count in summary["alpha_discovery_counts"].items():
        st.metric(display_value(decision), count)
    show_frame(
        st,
        summary["alpha_discovery_board"],
        "暂无候选决策面板数据。",
    )

    st.subheader("🧪 策略证据聚合")
    for decision, count in summary["strategy_counts"].items():
        st.metric(display_value(decision), count)
    show_frame(
        st,
        summary["strategy_evidence"],
        "暂无策略证据聚合数据。",
    )

    st.subheader("🚦 基线因子门控")
    for status, count in summary["counts"].items():
        st.metric(display_value(status), count)
    show_frame(st, summary["gates"], "暂无门控决策数据。")

    st.subheader("🧾 V5 候选质量")
    show_frame(
        st,
        summary["candidate_quality"],
        "暂无 V5 候选质量数据。",
    )
    show_frame(
        st,
        summary["candidate_outcomes"],
        "暂无 V5 候选后验汇总数据。",
    )
    show_warnings(st, summary["warnings"])
