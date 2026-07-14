from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from quant_lab.ai_research.contracts import (
    AI_PROMPT_VERSION,
    PROHIBITED_ACTIONS,
    AIResearchTask,
    EvidenceDocument,
    PriorResearchContext,
    TaskPreflight,
    canonical_json,
    compute_task_packet_sha256,
)

DEFAULT_MAX_MEMBER_BYTES = 256 * 1024
DEFAULT_MAX_DOCUMENT_CHARS = 40_000
DEFAULT_MAX_TOTAL_CHARS = 300_000
DEFAULT_MAX_CSV_ROWS = 64
DEFAULT_MAX_DOCS_PER_SECTION = 4

_ALLOWED_EXTENSIONS = {".json", ".md", ".csv", ".txt"}
_EXCLUDED_PARTS = {"__macosx", ".git", "secrets", "private", "restricted"}

# Exact and keyword rules deliberately favour already-generated summaries. The
# packet builder never scans the lake or triggers a research refresh.
_SECTION_RULES: dict[str, tuple[str, ...]] = {
    "core_state": (
        "manifest.json",
        "provenance.json",
        "data_quality.json",
        "executive_summary.md",
        "expert_questions.md",
        "system_acceptance_complete_status",
        "system_acceptance_dashboard",
    ),
    "factor_research": (
        "factor_definition",
        "factor_evidence",
        "factor_candidate",
        "factor_correlation",
        "factor_family_leaderboard",
        "factor_regime_effectiveness",
        "factor_forward_validation",
        "alpha_discovery_board",
        "alpha_factory",
        "research_promotion_decision",
    ),
    "strategy_evidence": (
        "strategy_evidence",
        "strategy_opportunity_advisory",
        "gate_effectiveness",
        "regime_strategy_advisory",
        "final_score_vs_alpha6_conflict",
        "bnb_strong_alpha6_bypass",
        "bottom_zone",
        "market_pressure",
    ),
    "cost_and_execution": (
        "cost_bucket",
        "cost_health",
        "cost_bootstrap",
        "cost_probe",
        "fill_bill",
        "slippage",
        "order_lifecycle",
        "trade_opportunity_funnel",
    ),
    "trade_learning": (
        "trade_outcome_attribution",
        "trade_learning_sample",
        "negative_expectancy",
        "false_block",
        "opportunity_cost",
        "decision_regret",
        "missed_opportunity",
        "entry_quality",
        "exit_quality",
        "premature",
        "gave_back_profit",
    ),
    "paper_lifecycle": (
        "paper_strategy_proposal",
        "paper_strategy_ack",
        "paper_strategy_tracker",
        "paper_strategy_registry",
        "paper_strategy_promotion",
        "paper_runtime_freshness",
        "paper_proposal_propagation",
        "paper_cohort",
        "paper_slippage_coverage",
    ),
    "operations": (
        "api_latency",
        "api_error",
        "api_auth",
        "lake_health",
        "no_trigger",
        "fallback_rate",
        "enforce_readiness",
        "completion_status",
    ),
}

_ALLOWED_FACTOR_TEMPLATES = [
    "feature",
    "neg_feature",
    "product",
    "difference",
    "safe_divide",
    "vol_adjusted",
    "range_vol_ratio",
    "range_location",
    "liquidity_adjusted",
]

_TASK_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def find_latest_expert_pack(exports_dir: str | Path) -> Path | None:
    root = Path(exports_dir)
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.rglob("*.zip")
        if path.is_file()
        and "expert" in path.name.lower()
        and not path.name.startswith(".")
    ]
    for candidate in sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    ):
        try:
            if zipfile.is_zipfile(candidate):
                with zipfile.ZipFile(candidate) as archive:
                    archive.infolist()
                return candidate
        except (OSError, zipfile.BadZipFile):
            continue
    return None


def build_ai_research_task(
    pack_path: str | Path,
    *,
    queue_root: str | Path,
    force: bool = False,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
    max_csv_rows: int = DEFAULT_MAX_CSV_ROWS,
    max_docs_per_section: int = DEFAULT_MAX_DOCS_PER_SECTION,
    now: datetime | None = None,
) -> tuple[AIResearchTask | None, Path | None]:
    pack = Path(pack_path)
    if not pack.is_file():
        raise FileNotFoundError(pack)
    if not zipfile.is_zipfile(pack):
        raise ValueError(f"AI research source pack must be a zip file: {pack}")

    queue = Path(queue_root)
    pending_root = queue / "pending"
    state_root = queue / "state"
    pending_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    state_path = state_root / "last_task.json"
    previous = _read_json_object(state_path)
    pack_stat = pack.stat()
    previous_task = str(previous.get("task_id") or "")
    same_file_fingerprint = (
        previous.get("source_pack_name") == pack.name
        and int(previous.get("source_pack_size_bytes") or -1) == pack_stat.st_size
        and int(previous.get("source_pack_mtime_ns") or -1) == pack_stat.st_mtime_ns
    )
    if not force and same_file_fingerprint and previous_task and _task_exists(queue, previous_task):
        return None, None

    pack_sha = _sha256_file(pack)
    if not force and previous.get("source_pack_sha256") == pack_sha:
        if previous_task and _task_exists(queue, previous_task):
            return None, None

    created = (now or datetime.now(UTC)).astimezone(UTC)
    task_id = _task_id(pack, pack_sha, created)
    sections: dict[str, list[EvidenceDocument]] = defaultdict(list)
    warnings: list[str] = []
    consumed_chars = 0

    with zipfile.ZipFile(pack) as archive:
        selected = _select_members(
            archive,
            max_docs_per_section=max_docs_per_section,
        )
        for section_name, members in selected.items():
            for member in members:
                if consumed_chars >= max_total_chars:
                    warnings.append("packet_total_character_limit_reached")
                    break
                document, document_warnings = _compact_member(
                    archive,
                    member,
                    max_member_bytes=max_member_bytes,
                    max_document_chars=min(max_document_chars, max_total_chars - consumed_chars),
                    max_csv_rows=max_csv_rows,
                )
                warnings.extend(document_warnings)
                if document is None:
                    continue
                encoded_length = len(canonical_json(document.model_dump(mode="json")))
                if consumed_chars + encoded_length > max_total_chars:
                    warnings.append(f"skipped_due_to_total_limit:{member.filename}")
                    continue
                sections[section_name].append(document)
                consumed_chars += encoded_length

    if not sections:
        raise ValueError(f"No supported evidence members found in expert pack: {pack}")

    preflight = _build_task_preflight(
        sections,
        checked_at=created,
        packet_warnings=warnings,
    )
    previous_research_context = _load_previous_research_context(
        state_root / "latest_research_context.json"
    )

    base_payload = {
        "schema_version": "quant_lab.ai_research_task.v1",
        "prompt_version": AI_PROMPT_VERSION,
        "task_id": task_id,
        "created_at": created.isoformat(),
        "source_pack_name": pack.name,
        "source_pack_sha256": pack_sha,
        "sections": {
            key: [item.model_dump(mode="json") for item in value]
            for key, value in sorted(sections.items())
        },
        "preflight": preflight.model_dump(mode="json"),
        "previous_research_context": (
            previous_research_context.model_dump(mode="json")
            if previous_research_context is not None
            else None
        ),
        "allowed_factor_templates": _ALLOWED_FACTOR_TEMPLATES,
        "prohibited_actions": list(PROHIBITED_ACTIONS),
        "warnings": sorted(set(warnings)),
    }
    provisional_task = AIResearchTask.model_validate(
        {**base_payload, "packet_sha256": "0" * 64}
    )
    packet_sha = compute_task_packet_sha256(provisional_task)
    task = provisional_task.model_copy(update={"packet_sha256": packet_sha})

    task_dir = pending_root / task_id
    if task_dir.exists():
        return None, None
    staging_root = queue / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{task_id}.", dir=staging_root))
    # mkdtemp always creates mode 0700. The NAS worker uses the queue-only
    # quantai group, so publish group-traversable directories before the
    # atomic pending rename.
    staging_dir.chmod(0o2770)
    try:
        _atomic_write_json(staging_dir / "task.json", task.model_dump(mode="json"))
        _atomic_write_json(
            staging_dir / "task_manifest.json",
            {
                "task_id": task_id,
                "source_pack_sha256": pack_sha,
                "packet_sha256": packet_sha,
                "published_at": created.isoformat(),
            },
        )
        try:
            os.replace(staging_dir, task_dir)
        except OSError:
            if task_dir.exists():
                return None, None
            raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    task_path = task_dir / "task.json"
    _atomic_write_json(
        state_path,
        {
            "task_id": task_id,
            "source_pack_name": pack.name,
            "source_pack_sha256": pack_sha,
            "source_pack_size_bytes": pack_stat.st_size,
            "source_pack_mtime_ns": pack_stat.st_mtime_ns,
            "packet_sha256": packet_sha,
            "created_at": created.isoformat(),
        },
    )
    return task, task_path


def _build_task_preflight(
    sections: dict[str, list[EvidenceDocument]],
    *,
    checked_at: datetime,
    packet_warnings: list[str],
) -> TaskPreflight:
    required_core_members = ["manifest.json", "provenance.json", "data_quality.json"]
    core_members = {
        document.source_member.lower().lstrip("./")
        for document in sections.get("core_state", [])
    }
    missing_core_members = [
        member for member in required_core_members if member not in core_members
    ]
    truncated_documents = [
        document
        for documents in sections.values()
        for document in documents
        if document.truncated
    ]
    truncated_core_members = sorted(
        document.source_member
        for document in sections.get("core_state", [])
        if document.truncated
    )
    blockers = [f"missing_core_member:{member}" for member in missing_core_members]
    blockers.extend(f"truncated_core_member:{member}" for member in truncated_core_members)
    warnings = sorted(set(packet_warnings))
    if truncated_documents and not truncated_core_members:
        warnings.append(f"truncated_non_core_documents:{len(truncated_documents)}")
    status: str
    if blockers:
        status = "BLOCK"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"
    return TaskPreflight(
        status=status,
        checked_at=checked_at,
        available_sections=sorted(sections),
        required_core_members=required_core_members,
        missing_core_members=missing_core_members,
        truncated_document_count=len(truncated_documents),
        blockers=sorted(set(blockers)),
        warnings=warnings,
    )


def _load_previous_research_context(path: Path) -> PriorResearchContext | None:
    payload = _read_json_object(path)
    if not payload:
        return None
    try:
        return PriorResearchContext.model_validate(payload)
    except (TypeError, ValueError):
        return None


def build_task_from_latest_export(
    exports_dir: str | Path,
    *,
    queue_root: str | Path,
    force: bool = False,
    **kwargs: Any,
) -> tuple[AIResearchTask | None, Path | None]:
    pack = find_latest_expert_pack(exports_dir)
    if pack is None:
        return None, None
    return build_ai_research_task(pack, queue_root=queue_root, force=force, **kwargs)


def queue_status(queue_root: str | Path) -> dict[str, Any]:
    root = Path(queue_root)
    counts = {}
    for state in ("pending", "running", "completed", "failed"):
        path = root / state
        counts[state] = sum(1 for item in path.iterdir() if item.is_dir()) if path.exists() else 0
    result_inbox = root / "results" / "inbox"
    result_imported = root / "results" / "imported"
    counts["result_inbox"] = (
        sum(1 for item in result_inbox.iterdir() if item.is_dir())
        if result_inbox.exists()
        else 0
    )
    counts["result_imported"] = (
        sum(1 for item in result_imported.iterdir() if item.is_dir())
        if result_imported.exists()
        else 0
    )
    return {
        "queue_root": str(root),
        "counts": counts,
        "last_task": _read_json_object(root / "state" / "last_task.json"),
    }


def _select_members(
    archive: zipfile.ZipFile,
    *,
    max_docs_per_section: int,
) -> dict[str, list[zipfile.ZipInfo]]:
    candidates = [info for info in archive.infolist() if _safe_supported_member(info)]
    selected: dict[str, list[zipfile.ZipInfo]] = {}
    used: set[str] = set()
    for section, keywords in _SECTION_RULES.items():
        ranked: list[tuple[int, str, zipfile.ZipInfo]] = []
        for info in candidates:
            lower = info.filename.lower()
            score = 0
            for keyword in keywords:
                if lower.endswith(keyword):
                    score = max(score, 100)
                elif keyword in lower:
                    score = max(score, 50)
            if score:
                # Prefer concise summary/json members over raw CSV when scores tie.
                suffix_bonus = {".md": 8, ".json": 7, ".csv": 5, ".txt": 3}.get(
                    PurePosixPath(lower).suffix,
                    0,
                )
                reports_bonus = 5 if lower.startswith("reports/") else 0
                root_core_bonus = 30 if lower in {
                    "manifest.json",
                    "provenance.json",
                    "data_quality.json",
                } else 0
                ranked.append(
                    (score + suffix_bonus + reports_bonus + root_core_bonus, lower, info)
                )
        section_items: list[zipfile.ZipInfo] = []
        for _score, _name, info in sorted(ranked, key=lambda item: (-item[0], item[1])):
            if info.filename in used:
                continue
            section_items.append(info)
            used.add(info.filename)
            if len(section_items) >= max_docs_per_section:
                break
        if section_items:
            selected[section] = section_items
    return selected


def _safe_supported_member(info: zipfile.ZipInfo) -> bool:
    if info.is_dir() or info.file_size <= 0:
        return False
    path = PurePosixPath(info.filename)
    if path.is_absolute() or ".." in path.parts:
        return False
    if any(part.lower() in _EXCLUDED_PARTS for part in path.parts):
        return False
    return path.suffix.lower() in _ALLOWED_EXTENSIONS


def _compact_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    *,
    max_member_bytes: int,
    max_document_chars: int,
    max_csv_rows: int,
) -> tuple[EvidenceDocument | None, list[str]]:
    warnings: list[str] = []
    with archive.open(member, "r") as stream:
        raw = stream.read(max_member_bytes + 1)
    truncated = len(raw) > max_member_bytes or member.file_size > max_member_bytes
    if len(raw) > max_member_bytes:
        raw = raw[:max_member_bytes]
    digest = hashlib.sha256(raw).hexdigest()
    suffix = PurePosixPath(member.filename).suffix.lower()
    text = raw.decode("utf-8", errors="replace")

    if suffix == ".csv":
        content, csv_truncated = _compact_csv(text, max_rows=max_csv_rows)
        source_format = "csv"
        truncated = truncated or csv_truncated
    elif suffix == ".json":
        content = _compact_json(text, max_chars=max_document_chars)
        source_format = "json"
    elif suffix == ".md":
        content = text[:max_document_chars]
        source_format = "markdown"
        truncated = truncated or len(text) > max_document_chars
    else:
        content = text[:max_document_chars]
        source_format = "text"
        truncated = truncated or len(text) > max_document_chars

    if len(canonical_json(content)) > max_document_chars:
        content = canonical_json(content)[:max_document_chars]
        truncated = True
        warnings.append(f"content_character_limit:{member.filename}")
    if truncated:
        warnings.append(f"truncated:{member.filename}")
    return (
        EvidenceDocument(
            source_member=member.filename,
            source_format=source_format,
            content_sha256=digest,
            source_size_bytes=member.file_size,
            truncated=truncated,
            content=content,
        ),
        warnings,
    )


def _compact_csv(text: str, *, max_rows: int) -> tuple[dict[str, Any], bool]:
    stream = io.StringIO(text)
    try:
        reader = csv.DictReader(stream)
        fieldnames = [str(item) for item in (reader.fieldnames or [])]
        rows: list[dict[str, str]] = []
        for index, row in enumerate(reader):
            if index >= max_rows:
                return {"columns": fieldnames, "rows": rows, "row_limit": max_rows}, True
            rows.append(
                {
                    str(key): _truncate_cell(value)
                    for key, value in row.items()
                    if key is not None
                }
            )
        return {"columns": fieldnames, "rows": rows, "row_limit": max_rows}, False
    except csv.Error:
        return {"columns": [], "rows": [], "raw_prefix": text[:20_000]}, True


def _compact_json(text: str, *, max_chars: int) -> Any:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text[:max_chars]
    return _truncate_json_value(value, budget=max_chars)


def _truncate_json_value(value: Any, *, budget: int, depth: int = 0) -> Any:
    if depth > 8:
        return "<depth_limit>"
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key in list(value)[:200]:
            output[str(key)] = _truncate_json_value(
                value[key],
                budget=max(100, budget // 4),
                depth=depth + 1,
            )
            if len(canonical_json(output)) >= budget:
                output["_truncated"] = True
                break
        return output
    if isinstance(value, list):
        output = []
        for item in value[:200]:
            output.append(_truncate_json_value(item, budget=max(100, budget // 4), depth=depth + 1))
            if len(canonical_json(output)) >= budget:
                output.append("<truncated>")
                break
        return output
    if isinstance(value, str):
        return value[:2000]
    return value


def _truncate_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= 1000 else text[:997] + "..."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_id(pack: Path, pack_sha: str, created: datetime) -> str:
    stem = _TASK_ID_RE.sub("-", pack.stem).strip("-.")[-64:] or "expert-pack"
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    return f"ai-{stamp}-{pack_sha[:12]}-{stem}"


def _task_exists(queue_root: Path, task_id: str) -> bool:
    return any(
        (queue_root / state / task_id).exists()
        for state in ("pending", "running", "completed", "failed")
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, 0o660)
        os.replace(temp_name, path)
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass
