from pathlib import Path
from typing import Any

import polars as pl


def streamlit_module(st_module: Any | None = None) -> Any:
    if st_module is not None:
        return st_module
    import streamlit as st

    return st


def show_warnings(st: Any, warnings: list[str]) -> None:
    for warning in warnings:
        st.warning(warning)


def show_frame(st: Any, df: pl.DataFrame, empty_message: str) -> None:
    if df.is_empty():
        st.info(empty_message)
        return
    st.dataframe(df)


def lake_caption(st: Any, lake_root: str | Path) -> None:
    st.caption(f"Lake root: {Path(lake_root)}")

