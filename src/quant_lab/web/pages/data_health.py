from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.data_health_summary(lake_root)

    st.title("Data Health")
    lake_caption(st, lake_root)
    st.metric("Duplicate bars", summary["duplicate_bar_count"])
    st.metric("Unclosed bars", summary["unclosed_bar_count"])
    st.metric("Schema violations", summary["schema_violation_count"])
    st.metric("Missing bar ratio", f"{summary['missing_bar_ratio']:.2%}")

    st.subheader("Latest Bar Per Symbol / Timeframe")
    show_frame(st, summary["latest_per_symbol"], "No market bars available.")

    st.subheader("Missing Bars")
    show_frame(st, summary["missing_bars"], "No missing bars detected.")

    st.subheader("Schema Violations")
    violations = pl.DataFrame({"violation": summary["schema_violations"]})
    show_frame(st, violations, "No schema violations detected.")

    st.subheader("Stale or Missing Datasets")
    show_frame(st, summary["stale_datasets"], "No stale dataset indicators detected.")
    show_warnings(st, summary["warnings"])

