from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.data_health_summary(lake_root)

    st.title("数据健康")
    lake_caption(st, lake_root)
    st.metric("重复 K 线数", summary["duplicate_bar_count"])
    st.metric("未闭合 K 线数", summary["unclosed_bar_count"])
    st.metric("结构违规数", summary["schema_violation_count"])
    st.metric("缺失 K 线比例", f"{summary['missing_bar_ratio']:.2%}")

    st.subheader("各标的 / 周期最新 K 线")
    show_frame(st, summary["latest_per_symbol"], "暂无 market_bar 数据。")

    st.subheader("缺失 K 线")
    show_frame(st, summary["missing_bars"], "未检测到缺失 K 线。")

    st.subheader("结构违规")
    violations = pl.DataFrame({"violation": summary["schema_violations"]})
    show_frame(st, violations, "未检测到结构违规。")

    st.subheader("过期或缺失的数据集")
    show_frame(st, summary["stale_datasets"], "未检测到过期或缺失的数据集。")
    show_warnings(st, summary["warnings"])
