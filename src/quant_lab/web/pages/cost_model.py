from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.cost_model_summary(lake_root)

    st.title("成本模型")
    lake_caption(st, lake_root)
    fallback_ratio = summary["fallback_ratio"]
    st.metric("真实成本行数", summary.get("actual_rows", 0))
    st.metric("代理成本行数", summary.get("proxy_rows", 0))
    st.metric("全局默认成本行数", summary.get("global_default_rows", 0))
    st.metric("回退比例", "不适用" if fallback_ratio is None else f"{fallback_ratio:.2%}")
    st.subheader("成本健康日报")
    show_frame(st, summary["cost_health"], "暂无 cost_health_daily 数据。")
    st.subheader("每日成本桶")
    show_frame(st, summary["costs"], "暂无 cost_bucket_daily 数据。")
    show_warnings(st, summary["warnings"])
