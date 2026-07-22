from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def render(_lake_root: str | Path, st_module: Any | None = None) -> None:
    st = _streamlit(st_module)
    url = os.environ.get("QUANT_LAB_BIGSCREEN_URL", "http://qyun2.hrhome.top:8027/web-v2")
    st.title("quant-lab Web V2 大屏")
    st.caption("只读研究战情室；不下单、不撤单、不修改交易端或交易所状态。")
    st.link_button("打开 V2 大屏", url)
    show_iframe(st, url, height=920, scrolling=False)


def show_iframe(st: Any, url: str, *, height: int, scrolling: bool) -> None:
    components = getattr(getattr(st, "components", None), "v1", None)
    component_iframe = getattr(components, "iframe", None)
    if callable(component_iframe):
        component_iframe(url, height=height, scrolling=scrolling)
        return

    iframe = getattr(st, "iframe", None)
    if callable(iframe):
        iframe(url, height=height)
        return

    raise AttributeError("Streamlit iframe API is unavailable")


def _streamlit(st_module: Any | None = None) -> Any:
    if st_module is not None:
        return st_module
    import streamlit as st

    return st
