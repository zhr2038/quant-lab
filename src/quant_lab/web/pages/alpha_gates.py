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

    st.title("Alpha 门控")
    lake_caption(st, lake_root)

    st.subheader("Alpha 候选决策面板")
    for decision, count in summary["alpha_discovery_counts"].items():
        st.metric(display_value(decision), count)
    show_frame(
        st,
        summary["alpha_discovery_board"],
        "暂无 alpha_discovery_board 候选决策数据。",
    )

    st.subheader("策略候选证据")
    for decision, count in summary["strategy_counts"].items():
        st.metric(display_value(decision), count)
    show_frame(
        st,
        summary["strategy_evidence"],
        "暂无 strategy_evidence 候选发现数据。",
    )

    st.subheader("策略候选证据样本")
    show_frame(
        st,
        summary["strategy_samples"],
        "暂无 strategy_evidence_sample 数据。",
    )

    st.subheader("V5 候选事件")
    show_frame(
        st,
        summary["candidate_events"],
        "暂无来自 candidate_snapshot.csv 的 v5_candidate_event 数据。",
    )

    st.subheader("V5 候选前向标签")
    show_frame(
        st,
        summary["candidate_labels"],
        "暂无 v5_candidate_label 数据。",
    )

    st.subheader("V5 候选后验汇总")
    show_frame(
        st,
        summary["candidate_outcomes"],
        "暂无 v5_candidate_outcome_summary 数据。",
    )

    st.subheader("V5 候选数据质量")
    show_frame(
        st,
        summary["candidate_quality"],
        "暂无 v5_candidate_quality_daily 数据。",
    )

    st.subheader("特征 Alpha 门控决策")
    for status, count in summary["counts"].items():
        st.metric(display_value(status), count)
    show_frame(st, summary["gates"], "暂无 gate_decision 数据。")
    show_warnings(st, summary["warnings"])
