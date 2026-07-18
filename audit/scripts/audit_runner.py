"""Hash-validated resumable entry point for the complete alpha audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.checkpoints import is_stage_complete, write_checkpoint  # noqa: E402
from audit.auditlib.paths import (  # noqa: E402
    ARTIFACTS,
    CHECKPOINTS,
    DATA_GOLD,
    DATA_SILVER,
    LOCAL_AUDIT_ROOT,
    REPORTS,
    latest_snapshot_id,
)

REPO = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)


STAGES = {
    "data": {
        "scripts": ["stage_silver.py", "stage_features.py"],
        "outputs": [
            DATA_SILVER / "bars_1h.parquet",
            DATA_SILVER / "funding.parquet",
            DATA_GOLD / "dynamic_universe.parquet",
            DATA_GOLD / "v5_raw_factors.parquet",
            ARTIFACTS / "data_quality_summary.csv",
            ARTIFACTS / "funding_quality_summary.csv",
            ARTIFACTS / "execution_data_quality_summary.csv",
            ARTIFACTS / "feature_manifest.json",
            ARTIFACTS / "signal_chain.json",
            REPORTS / "data_quality_report.md",
            REPORTS / "data_quality_dashboard.html",
            REPORTS / "current_signal_chain.md",
        ],
    },
    "leakage": {
        "scripts": ["stage_leakage.py"],
        "outputs": [ARTIFACTS / "leakage_test_results.csv", REPORTS / "data_leakage_audit.md"],
    },
    "factors": {
        "scripts": ["stage_factors.py", "stage_exposures.py", "stage_costs.py"],
        "outputs": [
            ARTIFACTS / "factor_validation_results.csv",
            ARTIFACTS / "factor_period_results.csv",
            ARTIFACTS / "factor_horizon_results.csv",
            ARTIFACTS / "factor_universe_results.csv",
            ARTIFACTS / "factor_symbol_contribution.csv",
            ARTIFACTS / "factor_exposure_results.csv",
            ARTIFACTS / "ic_statistics.csv",
            ARTIFACTS / "hac_statistics.csv",
            ARTIFACTS / "multiple_testing_summary.csv",
            ARTIFACTS / "negative_control_results.csv",
            ARTIFACTS / "cost_model_summary.csv",
            ARTIFACTS / "oos_lock.json",
            REPORTS / "statistical_methodology.md",
            REPORTS / "factor_validation_report.md",
            REPORTS / "cost_model_report.md",
        ],
    },
    "portfolio": {
        "scripts": ["stage_portfolio.py", "stage_capacity.py"],
        "outputs": [
            ARTIFACTS / "portfolio_backtest_results.csv",
            ARTIFACTS / "portfolio_timeseries.parquet",
            ARTIFACTS / "portfolio_symbol_contribution.csv",
            ARTIFACTS / "portfolio_benchmarks.csv",
            ARTIFACTS / "locked_portfolio_results.csv",
            ARTIFACTS / "portfolio_oos_lock.json",
            ARTIFACTS / "capacity_results.csv",
            REPORTS / "portfolio_validation_report.md",
            REPORTS / "capacity_report.md",
        ],
    },
    "report": {
        "scripts": ["stage_report.py"],
        "outputs": [
            ARTIFACTS / "final_decisions.json",
            ARTIFACTS / "run_manifest.json",
            REPORTS / "audit_dashboard.html",
            REPORTS / "executive_summary.md",
            REPORTS / "alpha_validity_audit.md",
            REPORTS / "production_impact_assessment.md",
            REPORTS / "resource_usage.md",
            REPORTS / "reproduction_guide.md",
        ],
    },
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _measure(command: list[str], stage: str) -> None:
    started = _utc_now()
    wall_start = time.perf_counter()
    disk_start = psutil.disk_usage(str(LOCAL_AUDIT_ROOT))
    process = subprocess.Popen(command, cwd=REPO)
    peak_rss = 0
    latest_user = latest_system = 0.0
    while process.poll() is None:
        try:
            root = psutil.Process(process.pid)
            processes = [root, *root.children(recursive=True)]
        except psutil.Error:
            processes = []
        rss = user = system = 0.0
        for item in processes:
            try:
                rss += item.memory_info().rss
                cpu = item.cpu_times()
                user += cpu.user
                system += cpu.system
            except psutil.Error:
                continue
        peak_rss = max(peak_rss, rss)
        latest_user = max(latest_user, user)
        latest_system = max(latest_system, system)
        time.sleep(0.10)
    return_code = process.wait()
    wall_seconds = time.perf_counter() - wall_start
    disk_end = psutil.disk_usage(str(LOCAL_AUDIT_ROOT))
    row = {
        "run_at": started,
        "stage": stage,
        "command": " ".join(command),
        "wall_seconds": wall_seconds,
        "cpu_user_seconds": latest_user,
        "cpu_system_seconds": latest_system,
        "peak_rss_mb": peak_rss / (1024 * 1024),
        "disk_used_before_gb": disk_start.used / (1024**3),
        "disk_used_after_gb": disk_end.used / (1024**3),
        "disk_delta_mb": (disk_end.used - disk_start.used) / (1024**2),
        "swap_events": 0,
        "return_code": return_code,
        "measurement_source": "psutil process-tree sampler",
    }
    path = ARTIFACTS / "resource_usage.csv"
    current = pl.read_csv(path) if path.is_file() else pl.DataFrame()
    updated = (
        pl.concat([current, pl.DataFrame([row])], how="diagonal_relaxed")
        if not current.is_empty()
        else pl.DataFrame([row])
    )
    updated.write_csv(path)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def run_stage(stage: str, *, resume: bool) -> None:
    if resume and is_stage_complete(LOCAL_AUDIT_ROOT, stage):
        print(f"SKIP {stage}: checkpoint hashes valid")
        return
    definition = STAGES[stage]
    started_at = _utc_now()
    command_texts: list[str] = []
    try:
        for script in definition["scripts"]:
            command = [str(PYTHON), str(REPO / "audit" / "scripts" / script)]
            command_texts.append(" ".join(command))
            print(f"RUN {stage}: {script}", flush=True)
            _measure(command, stage)
    except Exception as error:
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        failed = CHECKPOINTS / f"{stage}.failed.json"
        failed.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "started_at": started_at,
                    "failed_at": _utc_now(),
                    "git_commit": _git_commit(),
                    "snapshot_id": latest_snapshot_id(),
                    "commands": command_texts,
                    "status": "failed",
                    "error": repr(error),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    checkpoint = write_checkpoint(
        root=LOCAL_AUDIT_ROOT,
        stage=stage,
        started_at=started_at,
        git_commit=_git_commit(),
        snapshot_id=latest_snapshot_id(),
        command=" && ".join(command_texts),
        outputs=definition["outputs"],
    )
    print(f"DONE {stage}: {checkpoint}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", action="store_true", help="skip only hash-valid completed stages"
    )
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for stage in stages:
        run_stage(stage, resume=args.resume)


if __name__ == "__main__":
    main()
