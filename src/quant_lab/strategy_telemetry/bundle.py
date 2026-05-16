import hashlib
import posixpath
import re
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from quant_lab.strategy_telemetry.models import (
    BundleLimits,
    SafeExtractResult,
    V5BundleInspection,
    V5BundleValidationResult,
)

KNOWN_FILE_PATTERNS = [
    re.compile(r"^raw/recent_runs/[^/]+/decision_audit\.json$"),
    re.compile(r"^raw/recent_runs/[^/]+/summary\.json$"),
    re.compile(r"^raw/recent_runs/[^/]+/equity\.jsonl$"),
    re.compile(r"^raw/recent_runs/[^/]+/trades\.csv$"),
    re.compile(r"^raw/recent_runs/[^/]+/order_lifecycle\.csv$"),
    re.compile(r"^raw/state/(kill_switch|reconcile_status|ledger_status|auto_risk_eval)\.json$"),
    re.compile(r"^summaries/window_summary\.json$"),
    re.compile(r"^summaries/issues_to_fix\.json$"),
    re.compile(r"^raw/quant_lab/quant_lab_usage\.jsonl$"),
    re.compile(r"^raw/quant_lab/quant_lab_requests\.jsonl$"),
    re.compile(r"^raw/reports/quant_lab_usage\.jsonl$"),
    re.compile(r"^raw/reports/quant_lab_requests\.jsonl$"),
    re.compile(r"^reports/quant_lab_usage\.jsonl$"),
    re.compile(r"^reports/quant_lab_requests\.jsonl$"),
    re.compile(r"^summaries/quant_lab_compliance\.csv$"),
    re.compile(r"^summaries/quant_lab_cost_usage\.csv$"),
    re.compile(r"^summaries/quant_lab_fallbacks\.csv$"),
    re.compile(r"^summaries/router_decisions\.csv$"),
    re.compile(r"^summaries/trades_roundtrips\.csv$"),
    re.compile(r"^summaries/open_positions\.csv$"),
    re.compile(r"^summaries/high_score_blocked_targets\.csv$"),
    re.compile(r"^summaries/high_score_blocked_outcomes.*\.csv$"),
    re.compile(r"^summaries/btc_leadership_probe_blocked_outcomes.*\.csv$"),
    re.compile(r"^summaries/alt_impulse_shadow_outcomes.*\.csv$"),
    re.compile(r"^summaries/multi_position_swing_shadow_outcomes.*\.csv$"),
    re.compile(r"^summaries/factor_contribution_outcomes_by_factor.*\.csv$"),
    re.compile(r"^summaries/protect_sol_exception_shadow_outcomes.*\.csv$"),
    re.compile(r"^summaries/skipped_candidate_maturity_audit\.csv$"),
    re.compile(r"^raw/recent_runs/[^/]+/candidate_snapshot\.csv$"),
    re.compile(r"^reports/candidate_snapshot\.csv$"),
    re.compile(r"^summaries/candidate_snapshot\.csv$"),
    re.compile(r"^raw/reports/candidate_snapshot\.csv$"),
    re.compile(r"^reports/order_lifecycle\.csv$"),
    re.compile(r"^summaries/order_lifecycle\.csv$"),
    re.compile(r"^raw/reports/order_lifecycle\.csv$"),
    re.compile(r"^summaries/config_runtime_consumption_audit\.csv$"),
    re.compile(r"^raw/config_live_prod\.yaml$"),
    re.compile(r"^raw/reports/effective_live_config\.json$"),
    re.compile(r"^raw/logs/.*\.log$"),
]


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_v5_bundle(path: Path, limits: BundleLimits) -> V5BundleValidationResult:
    started_at = datetime.now(UTC)
    bundle_path = Path(path)
    reasons: list[str] = []
    warnings: list[str] = []
    detected: list[str] = []
    file_count = 0
    total_size = 0
    sha256: str | None = None
    size_bytes = 0

    if not bundle_path.exists():
        reasons.append("bundle file does not exist")
    elif bundle_path.suffixes[-2:] != [".tar", ".gz"] and bundle_path.suffix != ".tgz":
        reasons.append("bundle extension must be .tar.gz or .tgz")
    else:
        size_bytes = bundle_path.stat().st_size
        if size_bytes > limits.max_bundle_size_mb * 1024 * 1024:
            reasons.append("bundle exceeds max_bundle_size_mb")
        try:
            sha256 = compute_sha256(bundle_path)
            with tarfile.open(bundle_path, "r:gz") as archive:
                members = archive.getmembers()
                file_count = len(members)
                if file_count > limits.max_file_count:
                    reasons.append("bundle exceeds max_file_count")
                for member in members:
                    member_reasons = _dangerous_member_reasons(member)
                    reasons.extend(member_reasons)
                    if member.isfile():
                        total_size += member.size
                    if _is_detected_file(member.name):
                        detected.append(member.name)
                if total_size > limits.max_extracted_size_mb * 1024 * 1024:
                    reasons.append("bundle exceeds max_extracted_size_mb")
        except tarfile.TarError as exc:
            reasons.append(f"invalid tar.gz: {exc}")

    rejected = bool(reasons)
    finished_at = datetime.now(UTC)
    return V5BundleValidationResult(
        path=str(bundle_path),
        sha256=sha256,
        size_bytes=size_bytes,
        valid=not rejected,
        rejected=rejected,
        reasons=sorted(set(reasons)),
        warning_count=len(warnings),
        warnings=warnings,
        file_count=file_count,
        total_uncompressed_size_bytes=total_size,
        detected_files=sorted(set(detected)),
        started_at=started_at,
        finished_at=finished_at,
    )


def inspect_v5_bundle(path: Path) -> V5BundleInspection:
    limits = BundleLimits(
        max_bundle_size_mb=10_000,
        max_extracted_size_mb=50_000,
        max_file_count=200_000,
    )
    validation = validate_v5_bundle(path, limits)
    if validation.rejected or validation.sha256 is None:
        raise ValueError(f"cannot inspect invalid bundle: {validation.reasons}")
    return V5BundleInspection(
        path=str(path),
        sha256=validation.sha256,
        bundle_name=Path(path).name,
        bundle_ts=parse_bundle_ts(Path(path).name),
        detected_files=validation.detected_files,
        file_count=validation.file_count,
        total_uncompressed_size_bytes=validation.total_uncompressed_size_bytes,
    )


def safe_extract_v5_bundle(path: Path, target_dir: Path, limits: BundleLimits) -> SafeExtractResult:
    validation = validate_v5_bundle(path, limits)
    if validation.rejected:
        raise ValueError(f"unsafe V5 bundle rejected: {validation.reasons}")

    root = Path(target_dir).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    extracted: list[str] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                destination = (root / member.name).resolve()
                _ensure_within(root, destination)
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            destination = (root / member.name).resolve()
            _ensure_within(root, destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with destination.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            extracted.append(member.name)

    return SafeExtractResult(
        bundle_path=str(path),
        target_dir=str(root),
        extracted_files=sorted(extracted),
        file_count=len(extracted),
        total_uncompressed_size_bytes=validation.total_uncompressed_size_bytes,
    )


def parse_bundle_ts(bundle_name: str) -> datetime | None:
    match = re.search(r"(\d{8}T\d{6}Z)", bundle_name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def _dangerous_member_reasons(member: tarfile.TarInfo) -> list[str]:
    reasons: list[str] = []
    name = member.name
    if not name or not name.isprintable():
        reasons.append("bundle contains non-printable path")
    if "\\" in name:
        reasons.append(f"bundle contains non-portable path separator: {name}")
    if posixpath.isabs(name):
        reasons.append(f"bundle contains absolute path: {name}")
    if ".." in [part for part in re.split(r"[\\/]+", name) if part]:
        reasons.append(f"bundle contains path traversal: {name}")
    if member.issym():
        reasons.append(f"bundle contains symlink: {name}")
    if member.islnk():
        reasons.append(f"bundle contains hardlink: {name}")
    if member.isdev():
        reasons.append(f"bundle contains device file: {name}")
    if not (member.isfile() or member.isdir()):
        reasons.append(f"bundle contains unsupported member type: {name}")
    return reasons


def _is_detected_file(name: str) -> bool:
    logical_name = _logical_member_name(name)
    return any(pattern.match(logical_name) for pattern in KNOWN_FILE_PATTERNS)


def _logical_member_name(name: str) -> str:
    parts = [part for part in name.split("/") if part]
    if len(parts) > 1 and parts[0].startswith("v5_live_followup_bundle_"):
        return "/".join(parts[1:])
    return name


def _ensure_within(root: Path, destination: Path) -> None:
    if root != destination and root not in destination.parents:
        raise ValueError(f"bundle member escapes target directory: {destination}")
