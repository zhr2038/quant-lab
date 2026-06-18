from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.cost_model_summary(lake_root)

    st.title("成本模型")
    lake_caption(st, lake_root)
    hard_fallback_ratio = summary.get("hard_fallback_ratio")
    soft_fallback_ratio = summary.get("soft_fallback_ratio")
    st.metric("真实成本行数", summary.get("actual_rows", 0))
    st.metric("混合真实+代理成本行数", summary.get("mixed_rows", 0))
    st.metric("代理成本行数", summary.get("proxy_rows", 0))
    st.metric("全局默认成本行数", summary.get("global_default_rows", 0))
    st.metric(
        "硬回退比例",
        "不适用" if hard_fallback_ratio is None else f"{hard_fallback_ratio:.2%}",
    )
    st.metric(
        "软回退比例",
        "不适用" if soft_fallback_ratio is None else f"{soft_fallback_ratio:.2%}",
    )
    st.caption(
        "硬回退指 global_default、symbol 缺失、服务不可用或成本桶过期；"
        "软回退指样本不足、滑点未知或仅盘口代理。优先看硬回退，再看真实/混合成本覆盖。"
    )
    st.subheader("先看：成本健康结论")
    show_frame(st, summary["cost_health"], "暂无 cost_health_daily 数据。")
    st.subheader("成本冷启动 readiness")
    show_frame(
        st,
        summary["cost_bootstrap_readiness"],
        "暂无 cost_bootstrap_readiness 数据。",
    )
    st.subheader("各标的成本桶")
    st.caption(
        "每行重点看：成本来源、样本数、P75/往返 all-in 成本、以及 canary/scale 可信等级。"
    )
    show_frame(st, summary["costs"], "暂无 cost_bucket_daily 数据。")
    show_warnings(st, summary["warnings"])
