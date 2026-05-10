from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.alpha_gate_summary(lake_root)

    st.title("Alpha Gates")
    lake_caption(st, lake_root)
    st.subheader("Gate Decisions")
    show_frame(st, summary["gates"], "No gate decisions available.")
    show_warnings(st, summary["warnings"])

