from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.okx_collector_summary(lake_root)

    st.title("OKX Collectors")
    lake_caption(st, lake_root)
    st.metric("Trade print rows", summary["trade_print_rows"])
    st.metric("Orderbook snapshot rows", summary["orderbook_snapshot_rows"])

    st.subheader("Collector Counters")
    show_frame(st, summary["collectors"], "No collector counters available.")
    show_warnings(st, summary["warnings"])

