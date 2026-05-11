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
    st.metric("回退比例", "N/A" if fallback_ratio is None else f"{fallback_ratio:.2%}")
    st.subheader("cost_bucket_daily")
    show_frame(st, summary["costs"], "暂无 cost_bucket_daily 数据。")
    show_warnings(st, summary["warnings"])
