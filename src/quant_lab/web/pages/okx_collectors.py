from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.okx_collector_summary(lake_root)

    st.title("OKX 采集器")
    lake_caption(st, lake_root)
    st.metric("trade_print 行数", summary["trade_print_rows"])
    st.metric("orderbook_snapshot 行数", summary["orderbook_snapshot_rows"])

    st.subheader("采集器计数")
    show_frame(st, summary["collectors"], "暂无采集器计数。")
    st.subheader("采集器健康")
    show_frame(st, summary["collector_health"], "暂无 OKX 公共 WebSocket 采集器健康数据。")
    show_warnings(st, summary["warnings"])
