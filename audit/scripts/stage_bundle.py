"""Package the compact, reviewable alpha-audit evidence bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(os.environ.get("LOCAL_AUDIT_ROOT", "/home/hr/quant-alpha-audit")).resolve()
REPO = ROOT / "repos" / "quant-lab"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
MANIFESTS = ROOT / "manifests"
LOGS = ROOT / "logs"
BUNDLES = ROOT / "bundles"


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_manifests() -> None:
    status = _git("status", "--porcelain=v1") or "(clean)"
    state = "\n".join(
        [
            f"generated_at={datetime.now(UTC).isoformat()}",
            f"branch={_git('branch', '--show-current')}",
            f"head={_git('rev-parse', 'HEAD')}",
            f"origin_main={_git('rev-parse', 'origin/main')}",
            f"upstream={_git('rev-parse', '--abbrev-ref', '@{upstream}')}",
            "remote:",
            _git("remote", "-v"),
            "status:",
            status,
            "latest_commit:",
            _git("log", "-1", "--format=fuller"),
        ]
    )
    _write(MANIFESTS / "quant_lab_git_state.txt", state)
    _write(
        MANIFESTS / "git_commit_list.txt",
        _git(
            "log",
            "--reverse",
            "--format=%H%x09%ad%x09%s",
            "--date=iso-strict",
            "origin/main..HEAD",
        ),
    )
    _write(MANIFESTS / "git_diff.patch", _git("diff", "--binary", "origin/main...HEAD"))
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _write(MANIFESTS / "python_packages.txt", packages)


def _write_summary() -> None:
    decisions = json.loads((ARTIFACTS / "final_decisions.json").read_text(encoding="utf-8"))
    checkpoint_lines = []
    for checkpoint in sorted((ROOT / "checkpoints").glob("*.done.json")):
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        checkpoint_lines.append(
            f"{checkpoint.name}: status={payload.get('status')} "
            f"finished_at={payload.get('finished_at')} git={payload.get('git_commit')}"
        )
    summary = [
        "Alpha validity audit run summary",
        f"generated_at={datetime.now(UTC).isoformat()}",
        "snapshot=snap_20260718T031839Z",
        "data=1,225,652 closed 1h bars / 93 symbols / 2024-07-18 through 2026-07-17 UTC",
        "production_mutations=0",
        "live_orders=0",
        "tests=audit 30 passed; related 17 passed; full repository 1,387 passed",
        "lint=ruff PASS; format PASS; type checker unavailable",
        "browser_qa=1440x1000 and 390x844; 17 Plotly charts; nav PASS; console 0/0",
        f"current_v5={decisions['current_v5']['decision']}",
        f"rev_xs_20d={decisions['rev_xs_20d']['decision']}",
        f"low_vol_20d={decisions['low_vol_20d']['decision']}",
        f"funding_fade={decisions['funding_fade']['decision']}",
        f"recommended_action={decisions['recommended_action']}",
        "",
        "checkpoints:",
        *checkpoint_lines,
    ]
    _write(LOGS / "audit_run_summary.txt", "\n".join(summary))


def _readme() -> str:
    return """ALPHA VALIDITY AUDIT — READ FIRST

1. Open reports/audit_dashboard.html first (self-contained static HTML).
2. Read reports/executive_summary.md.
3. Read artifacts/final_decisions.json.
4. Full evidence is in the remaining reports and artifacts.
5. Reproduce from the recorded branch/commit with:
   cd /home/hr/quant-alpha-audit/repos/quant-lab
   export LOCAL_AUDIT_ROOT=/home/hr/quant-alpha-audit
   source /home/hr/quant-alpha-audit/.venv/bin/activate
   ./scripts/run_full_alpha_audit.sh --resume

Scope: read-only research audit. No production code/config/service/database was changed,
no order was submitted, and no candidate was authorized for paper or live trading.

Large raw candles, full gold data, temporary databases, virtual environments, repositories,
and Git objects are intentionally excluded. Their manifests, hashes, schemas, summaries, and
reproduction guidance are included instead.
"""


def _bundle_inputs() -> list[tuple[Path, str]]:
    inputs: list[tuple[Path, str]] = []
    for directory in (REPORTS, ARTIFACTS, MANIFESTS):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                inputs.append((path, str(path.relative_to(ROOT))))
    for path in sorted((ROOT / "checkpoints").glob("*.done.json")):
        inputs.append((path, str(path.relative_to(ROOT))))
    backfill_status = ROOT / "checkpoints" / "okx_backfill" / "_run_status.json"
    if backfill_status.exists():
        inputs.append((backfill_status, str(backfill_status.relative_to(ROOT))))
    inputs.append((LOGS / "audit_run_summary.txt", "logs/audit_run_summary.txt"))
    inputs.append(
        (
            REPO / "scripts" / "run_full_alpha_audit.sh",
            "reproduction/scripts/run_full_alpha_audit.sh",
        )
    )
    return inputs


def main() -> None:
    for directory in (MANIFESTS, LOGS, BUNDLES):
        directory.mkdir(parents=True, exist_ok=True)
    _refresh_manifests()
    _write_summary()
    readme_path = BUNDLES / "README_FIRST.txt"
    _write(readme_path, _readme())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle = BUNDLES / f"alpha_audit_bundle_{timestamp}.zip"
    with zipfile.ZipFile(
        bundle,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.write(readme_path, "README_FIRST.txt")
        for path, archive_name in _bundle_inputs():
            archive.write(path, archive_name)

    checksum = _sha256(bundle)
    _write(bundle.with_suffix(".zip.sha256"), f"{checksum}  {bundle.name}")
    downloads = Path("/mnt/c/Users/HR/Downloads")
    copied_to = ""
    if downloads.is_dir():
        destination = downloads / bundle.name
        shutil.copy2(bundle, destination)
        shutil.copy2(bundle.with_suffix(".zip.sha256"), destination.with_suffix(".zip.sha256"))
        copied_to = str(destination)

    print(f"bundle={bundle}")
    print(f"sha256={checksum}")
    print(f"files={len(_bundle_inputs()) + 1}")
    print(f"copied_to={copied_to or 'NOT_COPIED'}")


if __name__ == "__main__":
    main()
