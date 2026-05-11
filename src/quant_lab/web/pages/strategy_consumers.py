from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.strategy_consumer_summary(lake_root)

    st.title("策略消费者")
    lake_caption(st, lake_root)
    st.subheader("风险权限")
    show_frame(st, summary["permission_rows"], "暂无 risk_permission 数据。")

    st.subheader("回退信号")
    show_frame(st, summary["fallback_rows"], "decision_audit 中未找到回退行。")
    show_warnings(st, summary["warnings"])
