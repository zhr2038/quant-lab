from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.web import perf
from quant_lab.web.pages._common import lake_caption, show_frame, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    st.title("Web 性能")
    lake_caption(st, lake_root)

    events = perf.recent_events(limit=50)
    frame = pl.DataFrame(events) if events else pl.DataFrame()
    if not frame.is_empty():
        cache_hits = int(frame.get_column("cache_hit").sum()) if "cache_hit" in frame.columns else 0
        cache_misses = (
            int(frame.get_column("cache_miss").sum()) if "cache_miss" in frame.columns else 0
        )
        rglob_fallbacks = (
            int(frame.get_column("rglob_fallback").sum())
            if "rglob_fallback" in frame.columns
            else 0
        )
        st.metric("cache hit", cache_hits)
        st.metric("cache miss", cache_misses)
        st.metric("rglob fallback", rglob_fallbacks)

    st.subheader("最近 50 次 Web reader / render 事件")
    show_frame(st, frame, "暂无 Web 性能事件。")
