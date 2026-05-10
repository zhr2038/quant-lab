from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.market_regime_summary(lake_root)

    st.title("Market Regime")
    lake_caption(st, lake_root)

    st.subheader("Volatility Regime")
    show_frame(st, summary["regimes"], "No market regime data available.")

    st.subheader("Spread Bps")
    show_frame(st, summary["spread_bps"], "No orderbook spread data available.")

    st.subheader("Trade Activity")
    show_frame(st, summary["trade_activity"], "No trade activity data available.")

    st.subheader("Abnormal Symbols")
    show_frame(st, summary["abnormal_symbols"], "No abnormal symbols detected.")
    st.info("Funding rate and open interest are planned once those public datasets are published.")
    show_warnings(st, summary["warnings"])

