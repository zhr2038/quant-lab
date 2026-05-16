from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.export.daily import export_daily_pack
from quant_lab.reports.enforce_readiness import write_enforce_readiness_report
from quant_lab.research.alpha_discovery import build_and_publish_alpha_discovery_board
from quant_lab.research.candidate_labels import build_and_publish_candidate_labels
from quant_lab.research.strategy_evidence import build_and_publish_strategy_evidence
from quant_lab.risk.publish import publish_risk_permission
from quant_lab.strategy_telemetry.analyze import analyze_v5_telemetry
from quant_lab.time_display import beijing_today
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
    generated_pack = None
    if _button(st, "生成今日专家包", key="generate_today_expert_pack"):
        generated_pack = _generate_today_pack(st, lake_root=Path(lake_root), exports_root=root)
    _button(st, "刷新专家包列表", key="refresh_expert_pack_list")

    summary = readers.expert_export_summary(root)
    if generated_pack is not None:
        summary = _summary_preferring_generated_pack(root, generated_pack)

    st.subheader("专家包")
    show_frame(st, summary["packs"], "未找到专家包。")
    _render_pack_downloads(st, summary["packs"])

    st.subheader("清单摘要")
    manifest_rows = _dict_rows(summary["manifest_summary"])
    show_frame(st, manifest_rows, "暂无清单摘要。")

    st.subheader("数据质量摘要")
    quality_rows = _dict_rows(summary["data_quality_summary"])
    show_frame(st, quality_rows, "暂无数据质量摘要。")

    st.subheader("专家问题")
    questions = pl.DataFrame({"question": summary["expert_questions"]})
    show_frame(st, questions, "未找到专家问题。")
    show_warnings(st, summary["warnings"])


def _generate_today_pack(st: Any, *, lake_root: Path, exports_root: Path) -> Path | None:
    export_date = beijing_today().isoformat()
    exports_root.mkdir(parents=True, exist_ok=True)
    try:
        with _spinner(st, f"正在生成 {export_date} 专家包..."):
            refresh_warnings = _refresh_lake_before_export(lake_root, export_date=export_date)
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
                refresh_risk_permission=True,
                risk_strategy="v5",
                risk_version="5.0.0",
            )
        pack_path = Path(result.zip_path)
        pack_path.touch()
        _success(st, f"已生成专家包：{pack_path}")
        for warning in refresh_warnings:
            _warning(st, warning)
        return pack_path
    except Exception as exc:
        _error(st, f"生成专家包失败：{exc}")
        return None


def _refresh_lake_before_export(lake_root: Path, *, export_date: str) -> list[str]:
    warnings: list[str] = []
    steps = [
        (
            "analyze_v5_telemetry",
            lambda: analyze_v5_telemetry(lake_root=lake_root, date=export_date),
        ),
        (
            "build_candidate_labels",
            lambda: build_and_publish_candidate_labels(lake_root, as_of_date=export_date),
        ),
        (
            "build_strategy_evidence",
            lambda: build_and_publish_strategy_evidence(lake_root, as_of_date=export_date),
        ),
        (
            "build_alpha_discovery_board",
            lambda: build_and_publish_alpha_discovery_board(lake_root, as_of_date=export_date),
        ),
        (
            "publish_risk_permission",
            lambda: publish_risk_permission(lake_root, strategy="v5", version="5.0.0"),
        ),
        (
            "write_enforce_readiness_report",
            lambda: write_enforce_readiness_report(
                lake_root=lake_root,
                out_dir=readers.default_exports_root(lake_root),
                strategy="v5",
                version="5.0.0",
            ),
        ),
    ]
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            warnings.append(f"pre_export_refresh_failed:{name}:{exc}")
    return warnings


def _summary_preferring_generated_pack(exports_root: Path, generated_pack: Path) -> dict[str, Any]:
    summary = readers.expert_export_summary(exports_root)
    packs = summary.get("packs", pl.DataFrame())
    if packs.is_empty() or "path" not in packs.columns:
        return summary
    generated = str(generated_pack)
    if generated not in set(str(path) for path in packs["path"].to_list()):
        return summary
    ordered = pl.concat(
        [
            packs.filter(pl.col("path") == generated),
            packs.filter(pl.col("path") != generated).sort("modified_at", descending=True),
        ],
        how="vertical",
    )
    summary["latest_pack"] = generated
    summary["packs"] = ordered
    summary["manifest_summary"] = _read_json_member(generated_pack, "manifest.json")
    summary["data_quality_summary"] = _read_json_member(generated_pack, "data_quality.json")
    summary["expert_questions"] = [
        line
        for line in _read_text_member(generated_pack, "expert_questions.md").splitlines()
        if line.strip()
    ][:20]
    return summary


def _read_json_member(pack_path: Path, member: str) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(pack_path) as archive:
            return json.loads(archive.read(member).decode("utf-8"))
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
        return {}


def _read_text_member(pack_path: Path, member: str) -> str:
    try:
        with zipfile.ZipFile(pack_path) as archive:
            return archive.read(member).decode("utf-8")
    except (OSError, KeyError, UnicodeDecodeError, zipfile.BadZipFile):
        return ""


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


def _warning(st: Any, value: str) -> None:
    if hasattr(st, "warning"):
        st.warning(value)
    else:
        st.write(value)


def _dict_rows(values: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"key": key, "value": str(value)} for key, value in sorted(values.items())]
    )
