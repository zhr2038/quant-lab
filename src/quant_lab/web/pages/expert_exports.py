from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.web import readers
from quant_lab.web.pages._common import show_frame, show_warnings, streamlit_module


def render(
    lake_root: str | Path,
    st_module: Any | None = None,
    exports_root: str | Path | None = None,
) -> None:
    st = streamlit_module(st_module)
    root = (
        Path(exports_root)
        if exports_root is not None
        else readers.default_exports_root(lake_root)
    )
    summary = readers.expert_export_summary(root)

    st.title("Expert Exports")
    st.caption(f"Exports root: {root}")
    st.subheader("Expert Packs")
    show_frame(st, summary["packs"], "No expert packs found.")

    st.subheader("Manifest Summary")
    manifest_rows = _dict_rows(summary["manifest_summary"])
    show_frame(st, manifest_rows, "No manifest summary available.")

    st.subheader("Data Quality Summary")
    quality_rows = _dict_rows(summary["data_quality_summary"])
    show_frame(st, quality_rows, "No data quality summary available.")

    st.subheader("Expert Questions")
    questions = pl.DataFrame({"question": summary["expert_questions"]})
    show_frame(st, questions, "No expert questions found.")
    show_warnings(st, summary["warnings"])


def _dict_rows(values: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"key": key, "value": str(value)} for key, value in sorted(values.items())]
    )
