from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.export.daily import export_daily_pack
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

    st.title("专家包导出")
    st.caption(f"导出根目录：{root}")
    if _button(st, "生成今日专家包", key="generate_today_expert_pack"):
        _generate_today_pack(st, lake_root=Path(lake_root), exports_root=root)

    summary = readers.expert_export_summary(root)

    st.subheader("专家包")
    show_frame(st, summary["packs"], "未找到专家包。")
    _render_pack_downloads(st, summary["packs"])

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


def _generate_today_pack(st: Any, *, lake_root: Path, exports_root: Path) -> None:
    export_date = datetime.now(UTC).date().isoformat()
    try:
        with _spinner(st, f"正在生成 {export_date} 专家包..."):
            result = export_daily_pack(
                export_date=export_date,
                lake_root=lake_root,
                out_dir=exports_root,
                profile="expert",
                command_line=[
                    "qlab",
                    "export-daily",
                    "--date",
                    export_date,
                    "--lake-root",
                    str(lake_root),
                    "--out-dir",
                    str(exports_root),
                ],
            )
        _success(st, f"已生成专家包：{result.zip_path}")
    except Exception as exc:
        _error(st, f"生成专家包失败：{exc}")


def _render_pack_downloads(st: Any, packs: pl.DataFrame) -> None:
    if packs.is_empty() or "path" not in packs.columns or not hasattr(st, "download_button"):
        return
    st.subheader("下载")
    for row in packs.sort("modified_at", descending=True).head(20).to_dicts():
        path = Path(str(row["path"]))
        if not path.is_file():
            continue
        st.download_button(
            label=f"下载 {path.name}",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/zip",
            key=f"download-expert-pack-{path.name}",
        )


def _button(st: Any, label: str, **kwargs: Any) -> bool:
    if not hasattr(st, "button"):
        return False
    return bool(st.button(label, **kwargs))


class _NullSpinner:
    def __enter__(self) -> None:
        return None

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> bool:
        return False


def _spinner(st: Any, text: str) -> Any:
    if hasattr(st, "spinner"):
        return st.spinner(text)
    return _NullSpinner()


def _success(st: Any, value: str) -> None:
    if hasattr(st, "success"):
        st.success(value)
    else:
        st.write(value)


def _error(st: Any, value: str) -> None:
    if hasattr(st, "error"):
        st.error(value)
    else:
        st.warning(value)


def _dict_rows(values: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"key": key, "value": str(value)} for key, value in sorted(values.items())]
    )
