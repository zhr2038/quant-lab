from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
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
    export_date = beijing_today().isoformat()
    job_status = _poll_export_job(root, export_date)
    generated_pack = _consume_generated_pack(st)
    if generated_pack is None and job_status.get("state") == "succeeded":
        zip_path = Path(str(job_status.get("zip_path") or ""))
        if zip_path.is_file():
            generated_pack = zip_path
    _render_export_job_status(st, job_status)
    if generated_pack is not None:
        _success(st, f"已生成专家包：{generated_pack}")
    if _button(st, "生成今日专家包", key="generate_today_expert_pack"):
        if _web_background_export_enabled():
            started = _start_export_job(
                export_date=export_date,
                lake_root=Path(lake_root),
                exports_root=root,
            )
            _info(st, f"Expert pack generation started in background. PID={started.get('pid')}")
            if _rerun(st):
                return
        else:
            generated_pack = _generate_today_pack(
                st, lake_root=Path(lake_root), exports_root=root
            )
            if generated_pack is not None:
                _remember_generated_pack(st, generated_pack)
                if _rerun(st):
                    return
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


def _web_background_export_enabled() -> bool:
    value = os.environ.get("QUANT_LAB_WEB_EXPORT_BACKGROUND", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _render_export_job_status(st: Any, status: dict[str, Any]) -> None:
    state = str(status.get("state") or "")
    if state == "running":
        _info(
            st,
            "Expert pack generation is running in the background. "
            f"PID={status.get('pid')}; started_at={status.get('started_at')}",
        )
    elif state == "failed":
        _error(st, f"Expert pack generation failed: {status.get('error')}")


def _start_export_job(
    *,
    export_date: str,
    lake_root: Path,
    exports_root: Path,
) -> dict[str, Any]:
    exports_root.mkdir(parents=True, exist_ok=True)
    running = _poll_export_job(exports_root, export_date)
    if running.get("state") == "running":
        return running
    status_path = _export_job_status_path(exports_root, export_date)
    log_path = _export_job_log_path(exports_root, export_date)
    status = {
        "state": "starting",
        "export_date": export_date,
        "started_at": datetime.now(UTC).isoformat(),
        "status_path": str(status_path),
        "log_path": str(log_path),
    }
    _write_export_job_status(status_path, status)
    with log_path.open("ab") as log_file:
        popen_kwargs: dict[str, Any] = {
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "cwd": str(Path.cwd()),
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _export_job_script(),
                str(status_path),
                str(lake_root),
                str(exports_root),
                export_date,
            ],
            **popen_kwargs,
        )
    status["state"] = "running"
    status["pid"] = process.pid
    _write_export_job_status(status_path, status)
    return status


def _export_job_script() -> str:
    return r"""
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from quant_lab.export.daily import export_daily_pack

status_path = Path(sys.argv[1])
lake_root = Path(sys.argv[2])
exports_root = Path(sys.argv[3])
export_date = sys.argv[4]


def write_status(payload):
    tmp = status_path.with_suffix(status_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(status_path)


started_at = datetime.now(UTC).isoformat()
write_status({
    "state": "running",
    "export_date": export_date,
    "lake_root": str(lake_root),
    "exports_root": str(exports_root),
    "pid": os.getpid(),
    "started_at": started_at,
})
try:
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
    write_status({
        "state": "succeeded",
        "export_date": export_date,
        "lake_root": str(lake_root),
        "exports_root": str(exports_root),
        "zip_path": result.zip_path,
        "warnings": result.warnings,
        "pid": os.getpid(),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
    })
except Exception as exc:
    write_status({
        "state": "failed",
        "export_date": export_date,
        "lake_root": str(lake_root),
        "exports_root": str(exports_root),
        "error": f"{type(exc).__name__}: {exc}",
        "traceback_tail": traceback.format_exc()[-4000:],
        "pid": os.getpid(),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
    })
    raise
"""


def _poll_export_job(exports_root: Path, export_date: str) -> dict[str, Any]:
    status_path = _export_job_status_path(exports_root, export_date)
    status = _read_export_job_status(status_path)
    if status.get("state") != "running":
        return status
    pid = status.get("pid")
    if isinstance(pid, int) and _pid_is_running(pid):
        return status
    status = {
        **status,
        "state": "failed",
        "error": "export process exited before writing completion status",
        "finished_at": datetime.now(UTC).isoformat(),
    }
    _write_export_job_status(status_path, status)
    return status


def _read_export_job_status(status_path: Path) -> dict[str, Any]:
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_export_job_status(status_path: Path, status: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(status, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp_path.replace(status_path)


def _export_job_status_path(exports_root: Path, export_date: str) -> Path:
    return exports_root / f".quant_lab_web_export_{export_date}.json"


def _export_job_log_path(exports_root: Path, export_date: str) -> Path:
    return exports_root / f".quant_lab_web_export_{export_date}.log"


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                check=False,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return os.name == "nt"
    return True


def _generate_today_pack(st: Any, *, lake_root: Path, exports_root: Path) -> Path | None:
    export_date = beijing_today().isoformat()
    exports_root.mkdir(parents=True, exist_ok=True)
    try:
        with _spinner(st, f"正在生成 {export_date} 专家包..."):
            pack_path, refresh_warnings = _export_daily_from_web(
                export_date=export_date,
                lake_root=lake_root,
                exports_root=exports_root,
            )
        pack_path.touch()
        _success(st, f"已生成专家包：{pack_path}")
        for warning in refresh_warnings:
            _warning(st, warning)
        return pack_path
    except Exception as exc:
        _error(st, f"生成专家包失败：{exc}")
        return None


def _export_daily_from_web(
    *,
    export_date: str,
    lake_root: Path,
    exports_root: Path,
) -> tuple[Path, list[str]]:
    if os.environ.get("QUANT_LAB_WEB_EXPORT_MODE", "in_process") == "subprocess":
        return _export_daily_in_subprocess(
            export_date=export_date,
            lake_root=lake_root,
            exports_root=exports_root,
        )

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
    return Path(result.zip_path), list(result.warnings)


def _export_daily_in_subprocess(
    *,
    export_date: str,
    lake_root: Path,
    exports_root: Path,
) -> tuple[Path, list[str]]:
    script = (
        "import json, sys;"
        "from pathlib import Path;"
        "from quant_lab.export.daily import export_daily_pack;"
        "lake=Path(sys.argv[1]); out=Path(sys.argv[2]); day=sys.argv[3];"
        "result=export_daily_pack("
        "export_date=day, lake_root=lake, out_dir=out, profile='expert',"
        "command_line=['qlab','export-daily','--date',day,'--lake-root',str(lake),'--out-dir',str(out)],"
        "refresh_risk_permission=True, risk_strategy='v5', risk_version='5.0.0');"
        "print(json.dumps({'zip_path': result.zip_path, 'warnings': result.warnings}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(lake_root), str(exports_root), export_date],
        check=False,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("QUANT_LAB_WEB_EXPORT_TIMEOUT_SECONDS", "1800")),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"export subprocess failed with code {completed.returncode}: {detail[-2000:]}"
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("export subprocess did not return valid JSON") from exc
    return Path(str(payload["zip_path"])), list(payload.get("warnings") or [])


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
    # The caller may intentionally place a just-generated pack first even when
    # filesystem mtimes are tied or stale on a remote volume. Preserve that
    # order so the visible download list matches the success message.
    for row in packs.head(20).to_dicts():
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


def _info(st: Any, value: str) -> None:
    if hasattr(st, "info"):
        st.info(value)
    else:
        st.write(value)


def _remember_generated_pack(st: Any, pack_path: Path) -> None:
    session_state = getattr(st, "session_state", None)
    if session_state is None:
        return
    try:
        session_state["expert_exports_generated_pack"] = str(pack_path)
    except (TypeError, AttributeError):
        return


def _consume_generated_pack(st: Any) -> Path | None:
    session_state = getattr(st, "session_state", None)
    if session_state is None:
        return None
    try:
        value = session_state.pop("expert_exports_generated_pack", None)
    except (TypeError, AttributeError):
        return None
    return Path(str(value)) if value else None


def _rerun(st: Any) -> bool:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return True
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()
        return True
    return False


def _dict_rows(values: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"key": key, "value": str(value)} for key, value in sorted(values.items())]
    )
