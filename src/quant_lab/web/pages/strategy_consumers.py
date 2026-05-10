from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.strategy_consumer_summary(lake_root)

    st.title("Strategy Consumers")
    lake_caption(st, lake_root)
    st.subheader("Risk Permissions")
    show_frame(st, summary["permission_rows"], "No risk_permission data available.")

    st.subheader("Fallback Signals")
    show_frame(st, summary["fallback_rows"], "No fallback rows found in decision audits.")
    show_warnings(st, summary["warnings"])

