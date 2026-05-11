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

    st.title("专家包导出")
    st.caption(f"导出根目录：{root}")
    st.subheader("专家包")
    show_frame(st, summary["packs"], "未找到专家包。")

    st.subheader("Manifest 摘要")
    manifest_rows = _dict_rows(summary["manifest_summary"])
    show_frame(st, manifest_rows, "暂无 manifest 摘要。")

    st.subheader("数据质量摘要")
    quality_rows = _dict_rows(summary["data_quality_summary"])
    show_frame(st, quality_rows, "暂无数据质量摘要。")

    st.subheader("专家问题")
    questions = pl.DataFrame({"question": summary["expert_questions"]})
    show_frame(st, questions, "未找到专家问题。")
    show_warnings(st, summary["warnings"])


def _dict_rows(values: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"key": key, "value": str(value)} for key, value in sorted(values.items())]
    )
