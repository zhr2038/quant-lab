from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.market_regime_summary(lake_root)

    st.title("市场状态")
    lake_caption(st, lake_root)

    st.subheader("波动状态")
    show_frame(st, summary["regimes"], "暂无市场状态数据。")

    st.subheader("价差 bps")
    show_frame(st, summary["spread_bps"], "暂无订单簿价差数据。")

    st.subheader("成交活跃度")
    show_frame(st, summary["trade_activity"], "暂无成交活跃度数据。")

    st.subheader("异常标的")
    show_frame(st, summary["abnormal_symbols"], "未检测到异常标的。")
    st.info("资金费率和未平仓量会在对应公共数据集发布后加入。")
    show_warnings(st, summary["warnings"])
