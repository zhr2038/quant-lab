from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from quant_lab.export.daily import export_daily_pack
from quant_lab.time_display import beijing_today
from quant_lab.web import readers
from quant_lab.web.pages._common import show_frame, show_warnings, streamlit_module

WEB_EXPORT_MODE = "authoritative_snapshot"
DEFAULT_DOWNLOAD_PACK_LIMIT = 5
DEFAULT_DOWNLOAD_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_EXPORT_STATUS_STALE_SECONDS = 30 * 60
DEFAULT_REQUEST_FILE_PICKUP_STALE_SECONDS = 12 * 60
DEFAULT_EXPORT_REGENERATE_COOLDOWN_SECONDS = 3 * 60
DEFAULT_WEB_EXPORT_MEMORY_LIMIT_MB = 0
REQUEST_FILE_BACKGROUND_TRIGGER = "request_file"


def render(
    lake_root: str | Path,
    st_module: Any | None = None,
    exports_root: str | Path | None = None,
) -> None:
    st = streamlit_module(st_module)
    root = (
        Path(exports_root) if exports_root is not None else readers.default_exports_root(lake_root)
    )

    st.title("📦 专家包导出")
    st.caption(f"导出目录：{root}")
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
        if not _web_on_demand_export_enabled():
            generated_pack = _reuse_latest_today_pack(
                st,
                export_date=export_date,
                exports_root=root,
            )
            if generated_pack is not None:
                _remember_generated_pack(st, generated_pack)
                if _rerun(st):
                    return
        elif _web_background_export_enabled():
            started = _start_export_job(
                export_date=export_date,
                lake_root=Path(lake_root),
                exports_root=root,
            )
            if started.get("pid") is not None:
                _info(st, f"今日快照包已在后台生成中，进程 PID={started.get('pid')}")
            else:
                _info(
                    st,
                    "今日快照包请求已提交给后台导出 worker："
                    f"{started.get('systemd_unit') or started.get('trigger')}",
                )
            if _rerun(st):
                return
        else:
            generated_pack = _generate_today_pack(st, lake_root=Path(lake_root), exports_root=root)
            if generated_pack is not None:
                _remember_generated_pack(st, generated_pack)
                if _rerun(st):
                    return
    _button(st, "刷新专家包列表", key="refresh_expert_pack_list")

    summary = readers.expert_export_summary(root)
    if generated_pack is not None:
        summary = _summary_preferring_generated_pack(root, generated_pack)

    st.subheader("📥 可下载专家包")
    show_frame(st, summary["packs"], "未找到专家包。")
    _render_pack_downloads(st, summary["packs"])

    manifest_rows = _dict_rows(summary["manifest_summary"])
    if not manifest_rows.is_empty():
        st.subheader("🧾 清单摘要")
        show_frame(st, manifest_rows, "暂无清单摘要。")

    quality_rows = _dict_rows(summary["data_quality_summary"])
    if not quality_rows.is_empty():
        st.subheader("🩺 数据质量摘要")
        show_frame(st, quality_rows, "暂无数据质量摘要。")

    questions = pl.DataFrame({"question": summary["expert_questions"]})
    if not questions.is_empty():
        st.subheader("❓ 专家问题")
        show_frame(st, questions, "未找到专家问题。")
    show_warnings(st, summary["warnings"])


def _web_background_export_enabled() -> bool:
    value = os.environ.get("QUANT_LAB_WEB_EXPORT_BACKGROUND", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _web_on_demand_export_enabled() -> bool:
    value = os.environ.get("QUANT_LAB_WEB_ON_DEMAND_EXPORT", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _web_background_export_trigger() -> str:
    return os.environ.get("QUANT_LAB_WEB_EXPORT_BACKGROUND_TRIGGER", "subprocess").strip().lower()


def _render_export_job_status(st: Any, status: dict[str, Any]) -> None:
    state = str(status.get("state") or "")
    if state == "running":
        worker = (
            f"PID={status.get('pid')}"
            if status.get("pid") is not None
            else str(status.get("systemd_unit") or status.get("trigger") or "background")
        )
        _info(st, f"今日快照包正在后台生成。{worker}；开始时间={status.get('started_at')}")
    elif state == "failed":
        _error(st, f"专家包生成失败：{status.get('error')}")


def _start_export_job(
    *,
    export_date: str,
    lake_root: Path,
    exports_root: Path,
) -> dict[str, Any]:
    exports_root.mkdir(parents=True, exist_ok=True)
    running = _poll_export_job(exports_root, export_date)
    running_state = str(running.get("state") or "").lower()
    if running_state in {"starting", "running"} and not _export_job_is_stale(running):
        return running
    recent_success = _recent_success_export_status(
        running,
        exports_root=exports_root,
        export_date=export_date,
    )
    if recent_success is not None:
        return recent_success
    status_path = _export_job_status_path(exports_root, export_date)
    log_path = _export_job_log_path(exports_root, export_date)
    status = {
        "state": "starting",
        "mode": WEB_EXPORT_MODE,
        "export_date": export_date,
        "request_id": uuid4().hex,
        "started_at": datetime.now(UTC).isoformat(),
        "status_path": str(status_path),
        "log_path": str(log_path),
    }
    _write_export_job_status(status_path, status)
    if _web_background_export_trigger() == REQUEST_FILE_BACKGROUND_TRIGGER:
        return _start_export_request_file(
            status=status,
            status_path=status_path,
            export_date=export_date,
            lake_root=lake_root,
            exports_root=exports_root,
        )
    with log_path.open("wb") as log_file:
        popen_kwargs: dict[str, Any] = {
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "cwd": str(Path.cwd()),
            "env": _web_export_subprocess_env(),
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


def _start_export_request_file(
    *,
    status: dict[str, Any],
    status_path: Path,
    export_date: str,
    lake_root: Path,
    exports_root: Path,
) -> dict[str, Any]:
    request_path = _export_job_request_path(exports_root)
    request = {
        "request_id": status.get("request_id"),
        "export_date": export_date,
        "lake_root": str(lake_root),
        "exports_root": str(exports_root),
        "status_path": str(status_path),
        "requested_at": datetime.now(UTC).isoformat(),
        "trigger": REQUEST_FILE_BACKGROUND_TRIGGER,
    }
    tmp_path = request_path.with_suffix(request_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(request, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp_path.replace(request_path)
    status = {
        **status,
        "state": "running",
        "trigger": REQUEST_FILE_BACKGROUND_TRIGGER,
        "request_path": str(request_path),
        "systemd_unit": "quant-lab-web-export-request.service",
    }
    _write_export_job_status(status_path, status)
    return status


def _recent_success_export_status(
    status: dict[str, Any],
    *,
    exports_root: Path,
    export_date: str,
) -> dict[str, Any] | None:
    cooldown = _export_regenerate_cooldown_payload(
        status,
        exports_root=exports_root,
        export_date=export_date,
    )
    if int(cooldown.get("regenerate_cooldown_remaining_seconds") or 0) <= 0:
        return None
    return {
        **status,
        **cooldown,
        "state": "succeeded",
        "regenerate_skipped": True,
        "message": "recent expert pack generation reused; wait for cooldown before regenerating",
    }


def _export_regenerate_cooldown_payload(
    status: dict[str, Any],
    *,
    exports_root: Path,
    export_date: str,
) -> dict[str, Any]:
    cooldown_seconds = _web_export_regenerate_cooldown_seconds()
    base = {
        "regenerate_cooldown_seconds": cooldown_seconds,
        "regenerate_cooldown_remaining_seconds": 0,
        "regenerate_available_at": None,
        "regenerate_reuse_pack": None,
        "regenerate_reuse_pack_name": None,
    }
    if cooldown_seconds <= 0 or str(status.get("state") or "").lower() != "succeeded":
        return base
    pack_path = _pack_path_for_regenerate_cooldown(
        status,
        exports_root=exports_root,
        export_date=export_date,
    )
    if pack_path is None:
        return base
    completed_at = _completed_at_for_regenerate_cooldown(status, pack_path)
    if completed_at is None:
        return base
    age_seconds = (datetime.now(UTC) - completed_at).total_seconds()
    remaining = int(max(0, cooldown_seconds - age_seconds) + 0.999)
    if remaining <= 0:
        return base
    available_at = completed_at + timedelta(seconds=cooldown_seconds)
    return {
        **base,
        "regenerate_cooldown_remaining_seconds": remaining,
        "regenerate_available_at": available_at.isoformat(),
        "regenerate_reuse_pack": str(pack_path),
        "regenerate_reuse_pack_name": pack_path.name,
    }


def _pack_path_for_regenerate_cooldown(
    status: dict[str, Any],
    *,
    exports_root: Path,
    export_date: str,
) -> Path | None:
    root = exports_root.resolve()
    zip_path = status.get("zip_path")
    if zip_path:
        try:
            candidate = Path(str(zip_path)).resolve()
            candidate.relative_to(root)
        except (OSError, ValueError):
            candidate = None
        if (
            candidate is not None
            and candidate.is_file()
            and _pack_name_matches_export_date(candidate.name, export_date)
        ):
            return candidate
    return _latest_pack_for_export_date(exports_root, export_date)


def _completed_at_for_regenerate_cooldown(
    status: dict[str, Any],
    pack_path: Path,
) -> datetime | None:
    completed_at = _parse_status_datetime(status.get("finished_at"))
    if completed_at is not None:
        return completed_at
    try:
        return datetime.fromtimestamp(pack_path.stat().st_mtime, UTC)
    except OSError:
        return None


def _export_job_script() -> str:
    return r"""
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from quant_lab.export.daily import export_daily_pack

try:
    import resource
except ImportError:
    resource = None

status_path = Path(sys.argv[1])
lake_root = Path(sys.argv[2])
exports_root = Path(sys.argv[3])
export_date = sys.argv[4]


def write_status(payload):
    tmp = status_path.with_suffix(status_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(status_path)


started_at = datetime.now(UTC).isoformat()
memory_mb = int(os.environ.get("QUANT_LAB_WEB_EXPORT_MEMORY_LIMIT_MB", "0"))
if resource is not None and memory_mb > 0:
    limit_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
write_status({
    "state": "running",
    "mode": "authoritative_snapshot",
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
            "--no-refresh-risk-permission",
            "--pre-export-v5-refresh",
        ],
        refresh_risk_permission=False,
        risk_strategy="v5",
        risk_version="5.0.0",
        pre_export_v5_refresh=True,
        allow_stale_v5=False,
    )
    write_status({
        "state": "succeeded",
        "mode": "authoritative_snapshot",
        "export_date": export_date,
        "lake_root": str(lake_root),
        "exports_root": str(exports_root),
        "zip_path": str(result.zip_path),
        "warnings": result.warnings,
        "pid": os.getpid(),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
    })
except Exception as exc:
    write_status({
        "state": "failed",
        "mode": "authoritative_snapshot",
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


def _web_export_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["POLARS_MAX_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MALLOC_ARENA_MAX"] = "2"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _poll_export_job(exports_root: Path, export_date: str) -> dict[str, Any]:
    status_path = _export_job_status_path(exports_root, export_date)
    status = _read_export_job_status(status_path)
    if status.get("state") == "failed":
        recovered = _recover_failed_export_status_from_pack(
            exports_root,
            export_date,
            status_path,
            status,
        )
        if recovered is not None:
            return recovered
    status = _normalize_completed_export_status_times(status_path, status)
    if status.get("state") != "running":
        return status
    recovered = _recover_failed_export_status_from_pack(
        exports_root,
        export_date,
        status_path,
        status,
    )
    if recovered is not None:
        return recovered
    if _export_job_is_stale(status):
        status = {
            **status,
            "state": "failed",
            "error": "export process exceeded web status timeout; use scheduled export service",
            "finished_at": datetime.now(UTC).isoformat(),
        }
        recovered = _recover_failed_export_status_from_pack(
            exports_root,
            export_date,
            status_path,
            status,
        )
        if recovered is not None:
            return recovered
        _write_export_job_status(status_path, status)
        return status
    pid = status.get("worker_pid")
    if not isinstance(pid, int):
        pid = status.get("pid")
    if isinstance(pid, int) and _pid_is_running(pid):
        return status
    if status.get("trigger") == REQUEST_FILE_BACKGROUND_TRIGGER:
        status = _poll_request_file_export_job(
            exports_root=exports_root,
            export_date=export_date,
            status_path=status_path,
            status=status,
            pid=pid if isinstance(pid, int) else None,
        )
        return status
    status = {
        **status,
        "state": "failed",
        "error": "export process exited before writing completion status",
    }
    recovered = _recover_failed_export_status_from_pack(
        exports_root,
        export_date,
        status_path,
        status,
    )
    if recovered is not None:
        return recovered
    status["finished_at"] = datetime.now(UTC).isoformat()
    _write_export_job_status(status_path, status)
    return status


def _poll_request_file_export_job(
    *,
    exports_root: Path,
    export_date: str,
    status_path: Path,
    status: dict[str, Any],
    pid: int | None,
) -> dict[str, Any]:
    if pid is not None:
        failed = {
            **status,
            "state": "failed",
            "error": "systemd worker exited before writing completion status",
            "finished_at": datetime.now(UTC).isoformat(),
        }
        recovered = _recover_failed_export_status_from_pack(
            exports_root,
            export_date,
            status_path,
            failed,
        )
        if recovered is not None:
            return recovered
        _write_export_job_status(status_path, failed)
        return failed

    request_path = _request_path_from_status(status, exports_root)
    if request_path is not None and request_path.exists():
        request_age_seconds = _file_age_seconds(request_path)
        pickup_timeout_seconds = _positive_int_env(
            "QUANT_LAB_WEB_EXPORT_REQUEST_PICKUP_STALE_SECONDS",
            DEFAULT_REQUEST_FILE_PICKUP_STALE_SECONDS,
        )
        if request_age_seconds is None or request_age_seconds <= pickup_timeout_seconds:
            return status
        failed = {
            **status,
            "state": "failed",
            "error": (
                "systemd worker did not pick up web export request within "
                f"{pickup_timeout_seconds} seconds"
            ),
            "finished_at": datetime.now(UTC).isoformat(),
        }
        _write_export_job_status(status_path, failed)
        return failed

    failed = {
        **status,
        "state": "failed",
        "error": "systemd worker did not write completion status",
        "finished_at": datetime.now(UTC).isoformat(),
    }
    recovered = _recover_failed_export_status_from_pack(
        exports_root,
        export_date,
        status_path,
        failed,
    )
    if recovered is not None:
        return recovered
    _write_export_job_status(status_path, failed)
    return failed


def _recover_failed_export_status_from_pack(
    exports_root: Path,
    export_date: str,
    status_path: Path,
    status: dict[str, Any],
) -> dict[str, Any] | None:
    pack_path = _latest_pack_for_export_date(exports_root, export_date)
    if pack_path is None:
        return None
    finished_at = _parse_status_datetime(status.get("finished_at"))
    started_at = _parse_status_datetime(status.get("started_at"))
    try:
        pack_mtime = datetime.fromtimestamp(pack_path.stat().st_mtime, UTC)
    except OSError:
        return None
    if started_at is not None:
        if pack_mtime < started_at:
            return None
    elif finished_at is not None and pack_mtime <= finished_at:
        return None
    recovery_reason = "pack_created_after_export_job"
    recovered_started_at = pack_mtime if started_at is None else started_at
    recovered = {
        key: value
        for key, value in status.items()
        if key not in {"error", "traceback_tail"}
    } | {
        "state": "succeeded",
        "zip_path": str(pack_path),
        "recovered_from_failed_status": True,
        "recovery_reason": recovery_reason,
        "recovered_pack_mtime": pack_mtime.isoformat(),
        "started_at": recovered_started_at.isoformat(),
        "finished_at": pack_mtime.isoformat(),
    }
    _write_export_job_status(status_path, recovered)
    return recovered


def _normalize_completed_export_status_times(
    status_path: Path,
    status: dict[str, Any],
) -> dict[str, Any]:
    if status.get("state") != "succeeded":
        return status
    started_at = _parse_status_datetime(status.get("started_at"))
    finished_at = _parse_status_datetime(status.get("finished_at"))
    if started_at is None or finished_at is None or started_at <= finished_at:
        return status
    normalized = {
        **status,
        "started_at": finished_at.isoformat(),
        "timestamp_normalized": True,
        "timestamp_normalization_reason": "started_at_after_finished_at",
    }
    _write_export_job_status(status_path, normalized)
    return normalized


def _export_job_is_stale(status: dict[str, Any]) -> bool:
    started_at = status.get("started_at")
    if not started_at:
        return False
    started = _parse_status_datetime(started_at)
    if started is None:
        return False
    timeout_seconds = _positive_int_env(
        "QUANT_LAB_WEB_EXPORT_STATUS_STALE_SECONDS",
        DEFAULT_EXPORT_STATUS_STALE_SECONDS,
    )
    age_seconds = (datetime.now(UTC) - started.astimezone(UTC)).total_seconds()
    return age_seconds > timeout_seconds


def _request_path_from_status(status: dict[str, Any], exports_root: Path) -> Path | None:
    path_text = status.get("request_path")
    if not path_text:
        return _export_job_request_path(exports_root)
    try:
        return Path(str(path_text))
    except OSError:
        return None


def _file_age_seconds(path: Path) -> float | None:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:
        return None
    return (datetime.now(UTC) - mtime).total_seconds()


def _parse_status_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)



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


def _export_job_request_path(exports_root: Path) -> Path:
    return exports_root / ".quant_lab_web_export_request.json"


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
    proc_state = _linux_proc_state(pid)
    if proc_state == "Z":
        return False
    if proc_state is not None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return os.name == "nt"
    return True


def _linux_proc_state(pid: int) -> str | None:
    if os.name == "nt":
        return None
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        after_name = stat_text.rsplit(")", maxsplit=1)[1].strip()
    except IndexError:
        return None
    return after_name.split(maxsplit=1)[0] if after_name else None


def _generate_today_pack(st: Any, *, lake_root: Path, exports_root: Path) -> Path | None:
    export_date = beijing_today().isoformat()
    exports_root.mkdir(parents=True, exist_ok=True)
    if not _web_on_demand_export_enabled():
        return _reuse_latest_today_pack(st, export_date=export_date, exports_root=exports_root)
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


def _reuse_latest_today_pack(st: Any, *, export_date: str, exports_root: Path) -> Path | None:
    pack_path = _latest_pack_for_export_date(exports_root, export_date)
    if pack_path is None:
        _warning(
            st,
            "今日专家包尚未由定时任务生成。为保护 qyun2，Web 不直接运行重计算；"
            "请等待 quant-lab-daily-export.service，或在维护窗口运行 qlab export-daily。",
        )
        return None
    _success(st, f"已找到今日专家包，可直接下载：{pack_path}")
    return pack_path


def _latest_pack_for_export_date(exports_root: Path, export_date: str) -> Path | None:
    if not exports_root.exists():
        return None
    packs = [
        path
        for path in exports_root.glob("quant_lab_expert_pack_*.zip")
        if path.is_file()
        and _pack_name_matches_export_date(path.name, export_date)
    ]
    if not packs:
        return None
    packs.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    authoritative = [path for path in packs if _expert_pack_authoritative_snapshot(path)]
    return authoritative[0] if authoritative else packs[0]


def _pack_name_matches_export_date(file_name: str, export_date: str) -> bool:
    if file_name == f"quant_lab_expert_pack_{export_date}.zip" or file_name.startswith(
        f"quant_lab_expert_pack_{export_date}_"
    ):
        return True
    target_day = str(export_date or "").replace("-", "")
    match = re.match(
        r"^quant_lab_expert_pack_\d{4}-\d{2}-\d{2}_(\d{8})T\d{6}",
        str(file_name or ""),
    )
    return bool(target_day and match and match.group(1) == target_day)


def _expert_pack_authoritative_snapshot(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    except Exception:
        return False
    return isinstance(manifest, dict) and bool(manifest.get("authoritative_snapshot"))


def _export_daily_from_web(
    *,
    export_date: str,
    lake_root: Path,
    exports_root: Path,
) -> tuple[Path, list[str]]:
    if os.environ.get("QUANT_LAB_WEB_EXPORT_MODE", "subprocess") == "subprocess":
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
            "--no-refresh-risk-permission",
            "--pre-export-v5-refresh",
        ],
        refresh_risk_permission=False,
        risk_strategy="v5",
        risk_version="5.0.0",
        pre_export_v5_refresh=True,
        allow_stale_v5=False,
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
        "command_line=['qlab','export-daily','--date',day,'--lake-root',str(lake),'--out-dir',str(out),"
        "'--no-refresh-risk-permission','--pre-export-v5-refresh'],"
        "refresh_risk_permission=False, risk_strategy='v5', risk_version='5.0.0',"
        "pre_export_v5_refresh=True, allow_stale_v5=False);"
        "print(json.dumps({'zip_path': str(result.zip_path), 'warnings': result.warnings}))"
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
        raise RuntimeError(f"专家包导出子进程失败，退出码 {completed.returncode}: {detail[-2000:]}")
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("专家包导出子进程未返回有效 JSON") from exc
    return Path(str(payload["zip_path"])), list(payload.get("warnings") or [])


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
    limit = _download_pack_limit()
    max_bytes = _download_max_bytes()
    # The caller may intentionally place a just-generated pack first even when
    # filesystem mtimes are tied or stale on a remote volume. Preserve that
    # order so the visible download list matches the success message.
    rendered = 0
    too_large: list[str] = []
    for row in packs.to_dicts():
        if rendered >= limit:
            break
        path = Path(str(row["path"]))
        if not path.is_file():
            continue
        stat = path.stat()
        if stat.st_size > max_bytes:
            too_large.append(f"{path.name} ({_format_bytes(stat.st_size)})")
            continue
        st.download_button(
            label=f"下载 {path.name} ({_format_bytes(stat.st_size)})",
            data=_download_pack_bytes(str(path), stat.st_size, stat.st_mtime_ns),
            file_name=path.name,
            mime="application/zip",
            key=f"download-expert-pack-{path.name}",
        )
        rendered += 1
    if packs.height > rendered:
        _caption(
            st,
            f"直接下载按钮仅加载最近 {rendered} 个可下载专家包；完整历史仍在上方列表中。",
        )
    if too_large:
        _warning(
            st,
            "以下专家包超过 Web 直接下载上限，请从服务器路径下载：" + "；".join(too_large[:5]),
        )


@lru_cache(maxsize=8)
def _download_pack_bytes(path: str, size_bytes: int, mtime_ns: int) -> bytes:
    _ = (size_bytes, mtime_ns)
    return Path(path).read_bytes()


def _download_pack_limit() -> int:
    return _positive_int_env("QUANT_LAB_WEB_DOWNLOAD_MAX_PACKS", DEFAULT_DOWNLOAD_PACK_LIMIT)


def _download_max_bytes() -> int:
    value = _positive_int_env("QUANT_LAB_WEB_DOWNLOAD_MAX_MB", 128)
    return value * 1024 * 1024


def _web_export_regenerate_cooldown_seconds() -> int:
    try:
        value = int(
            os.environ.get(
                "QUANT_LAB_WEB_EXPORT_REGENERATE_COOLDOWN_SECONDS",
                str(DEFAULT_EXPORT_REGENERATE_COOLDOWN_SECONDS),
            )
        )
    except ValueError:
        return DEFAULT_EXPORT_REGENERATE_COOLDOWN_SECONDS
    return max(0, value)


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


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


def _caption(st: Any, value: str) -> None:
    if hasattr(st, "caption"):
        st.caption(value)
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
