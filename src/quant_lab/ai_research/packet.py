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
CORE_JSON_MAX_MEMBER_BYTES = 2 * 1024 * 1024
RESEARCH_CSV_MAX_MEMBER_BYTES = 16 * 1024 * 1024
FACTOR_AUDIT_MAX_DOCUMENT_CHARS = 80_000

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
_CORE_SUMMARY_MEMBERS = {"manifest.json", "data_quality.json"}
_DETERMINISTIC_CSV_SUMMARY_MEMBERS = {"reports/alpha_discovery_board.csv"}
_ALPHA_FACTORY_AUDIT_MEMBERS = {
    "reports/alpha_factory_candidates.csv",
    "reports/alpha_factory_results.csv",
    "reports/alpha_factory_promotion_queue.csv",
}
_FACTOR_VALIDATION_AUDIT_MEMBERS = {
    "reports/factor_definitions.csv",
    "reports/factor_dedupe_decision.csv",
    "reports/factor_forward_validation.csv",
}
_RESEARCH_DATASET_KEYWORDS = (
    "alpha",
    "factor",
    "paper",
    "cost",
    "risk",
    "permission",
    "proposal",
    "tracker",
    "opportunity",
    "trade",
    "v5",
)


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
        audit_documents, audit_sources, audit_warnings = (
            _build_factor_research_audit_documents(archive)
        )
        warnings.extend(audit_warnings)
        for document in audit_documents:
            encoded_length = len(canonical_json(document.model_dump(mode="json")))
            if consumed_chars + encoded_length > max_total_chars:
                warnings.append(
                    f"skipped_due_to_total_limit:{document.source_member}"
                )
                continue
            sections["factor_research"].append(document)
            consumed_chars += encoded_length

        selected = _select_members(
            archive,
            max_docs_per_section=max_docs_per_section,
        )
        if audit_documents:
            # The derived documents already carry every current Alpha Factory and
            # factor-validation row. Keep the large discovery board summary for
            # population context, but do not spend the packet budget duplicating
            # its component CSVs or terse Markdown summaries.
            selected["factor_research"] = [
                member
                for member in selected.get("factor_research", [])
                if member.filename.lower().lstrip("./")
                == "reports/alpha_discovery_board.csv"
            ]
        for section_name, members in selected.items():
            for member in members:
                if member.filename.lower().lstrip("./") in audit_sources:
                    continue
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
    warnings = sorted(set(packet_warnings))
    warnings.extend(f"truncated_core_member:{member}" for member in truncated_core_members)
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
    normalized_name = member.filename.lower().lstrip("./")
    deterministic_core_summary = normalized_name in _CORE_SUMMARY_MEMBERS
    deterministic_csv_summary = normalized_name in _DETERMINISTIC_CSV_SUMMARY_MEMBERS
    if deterministic_core_summary:
        read_limit = max(max_member_bytes, CORE_JSON_MAX_MEMBER_BYTES)
    elif deterministic_csv_summary:
        read_limit = max(max_member_bytes, RESEARCH_CSV_MAX_MEMBER_BYTES)
    else:
        read_limit = max_member_bytes
    with archive.open(member, "r") as stream:
        raw = stream.read(read_limit + 1)
    truncated = len(raw) > read_limit or member.file_size > read_limit
    if len(raw) > read_limit:
        raw = raw[:read_limit]
    digest = hashlib.sha256(raw).hexdigest()
    suffix = PurePosixPath(member.filename).suffix.lower()
    text = raw.decode("utf-8", errors="replace")
    representation = "full"

    if suffix == ".csv":
        if deterministic_csv_summary and not truncated:
            content, summary_complete = _summarize_research_csv(
                text,
                max_rows=max_csv_rows,
                max_chars=max_document_chars,
            )
            representation = (
                "deterministic_summary" if summary_complete else "truncated_prefix"
            )
            csv_truncated = not summary_complete
        else:
            content, csv_truncated = _compact_csv(text, max_rows=max_csv_rows)
        source_format = "csv"
        truncated = truncated or csv_truncated
    elif suffix == ".json":
        if deterministic_core_summary and not truncated:
            content, summary_complete = _compact_core_json(
                text,
                source_member=normalized_name,
                max_chars=max_document_chars,
            )
            representation = (
                "deterministic_summary" if summary_complete else "truncated_prefix"
            )
            truncated = not summary_complete
        else:
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
        representation = "truncated_prefix"
        warnings.append(f"content_character_limit:{member.filename}")
    if truncated:
        if representation == "full":
            representation = "truncated_prefix"
        warnings.append(f"truncated:{member.filename}")
    return (
        EvidenceDocument(
            source_member=member.filename,
            source_format=source_format,
            content_sha256=digest,
            source_size_bytes=member.file_size,
            representation=representation,
            truncated=truncated,
            content=content,
        ),
        warnings,
    )


def _compact_core_json(
    text: str,
    *,
    source_member: str,
    max_chars: int,
) -> tuple[Any, bool]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text[:max_chars], False
    if not isinstance(value, dict):
        content = _truncate_json_value(value, budget=max_chars)
        return content, len(canonical_json(content)) <= max_chars

    if source_member == "manifest.json":
        content = _summarize_manifest(value)
    elif source_member == "data_quality.json":
        content = _summarize_data_quality(value)
    else:
        content = _truncate_json_value(value, budget=max_chars)
    return content, len(canonical_json(content)) <= max_chars


def _summarize_manifest(value: dict[str, Any]) -> dict[str, Any]:
    summary = _scalar_items(value)
    row_counts = value.get("row_counts")
    if isinstance(row_counts, dict):
        summary["row_counts"] = row_counts
    summary["dataset_freshness"] = _summarize_dataset_mapping(
        value.get("dataset_freshness"),
        max_entries=72,
    )
    for key in ("acceptance_set_sha256_relationship", "github_ci_status", "lake_file_health"):
        if key in value:
            summary[key] = _truncate_json_value(value[key], budget=4_000)
    files = value.get("files")
    sections = value.get("sections")
    summary["_representation"] = {
        "kind": "deterministic_summary",
        "source_key_count": len(value),
        "file_count": len(files) if isinstance(files, list) else 0,
        "section_count": len(sections) if isinstance(sections, (list, dict)) else 0,
        "dataset_freshness_count": (
            len(value.get("dataset_freshness", {}))
            if isinstance(value.get("dataset_freshness"), dict)
            else 0
        ),
        "selection_rule": "all_scalars_all_row_counts_and_research_relevant_or_non_ok_freshness",
    }
    return summary


def _summarize_data_quality(value: dict[str, Any]) -> dict[str, Any]:
    summary = _scalar_items(value)
    for key in ("failures", "warnings"):
        if key in value:
            summary[key] = _truncate_json_value(value[key], budget=4_000)
    summary["checks"] = _summarize_check_list(value.get("checks"), max_entries=12)
    for key in ("dataset_governance", "registry_quality"):
        nested = value.get(key)
        if isinstance(nested, dict):
            compact = _scalar_items(nested)
            compact["checks"] = _summarize_check_list(
                nested.get("checks"),
                max_entries=16,
            )
            summary[key] = compact
    nested_budgets = {
        "decision_audit": 2_000,
        "quant_lab_enforce_readiness": 4_000,
        "risk_permission": 4_000,
        "v5_pre_export": 4_000,
    }
    for key, budget in nested_budgets.items():
        if key in value:
            summary[key] = _truncate_json_value(value[key], budget=budget)
    summary["_representation"] = {
        "kind": "deterministic_summary",
        "source_key_count": len(value),
        "selection_rule": "all_scalars_all_failures_and_warnings_plus_bounded_non_ok_first_checks",
    }
    return summary


def _scalar_items(value: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): item
        for key, item in value.items()
        if item is None or isinstance(item, (str, int, float, bool))
    }


def _summarize_dataset_mapping(value: Any, *, max_entries: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"count": 0, "selected": {}}
    ranked = sorted(
        value.items(),
        key=lambda item: (
            0 if _record_is_non_ok(item[1]) else 1,
            0
            if any(
                keyword in str(item[0]).lower()
                for keyword in _RESEARCH_DATASET_KEYWORDS
            )
            else 1,
            str(item[0]),
        ),
    )
    return {
        "count": len(value),
        "selected": {
            str(key): _truncate_json_value(item, budget=600)
            for key, item in ranked[:max_entries]
        },
    }


def _summarize_check_list(value: Any, *, max_entries: int) -> dict[str, Any]:
    if not isinstance(value, list):
        return {"count": 0, "selected": []}
    non_ok = [item for item in value if _record_is_non_ok(item)]
    ok = [item for item in value if not _record_is_non_ok(item)]
    selected = (non_ok + ok)[:max_entries]
    return {
        "count": len(value),
        "non_ok_count": len(non_ok),
        "selected": [_truncate_json_value(item, budget=400) for item in selected],
    }


def _record_is_non_ok(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    outcome = " ".join(
        str(value.get(key) or "").upper()
        for key in ("status", "state", "verdict", "result")
    ).strip()
    if outcome:
        if any(token in outcome for token in ("PASS", "OK", "HEALTHY", "READY")):
            return False
        return any(
            token in outcome
            for token in (
                "WARN",
                "FAIL",
                "ERROR",
                "CRITICAL",
                "BLOCK",
                "STALE",
                "MISSING",
            )
        )
    severity = str(value.get("severity") or "").upper()
    return any(token in severity for token in ("WARN", "ERROR", "CRITICAL"))


def _build_factor_research_audit_documents(
    archive: zipfile.ZipFile,
) -> tuple[list[EvidenceDocument], set[str], list[str]]:
    available = {
        info.filename.lower().lstrip("./"): info
        for info in archive.infolist()
        if _safe_supported_member(info)
    }
    documents: list[EvidenceDocument] = []
    consumed_sources: set[str] = set()
    warnings: list[str] = []

    alpha_sources = {
        name: available[name]
        for name in _ALPHA_FACTORY_AUDIT_MEMBERS
        if name in available
    }
    if len(alpha_sources) == len(_ALPHA_FACTORY_AUDIT_MEMBERS):
        content, complete, detail_warnings = _build_alpha_factory_candidate_audit(
            archive,
            alpha_sources,
        )
        warnings.extend(detail_warnings)
        document = _derived_evidence_document(
            "derived/alpha_factory_candidate_audit.json",
            content,
            alpha_sources.values(),
            complete=complete,
        )
        documents.append(document)
        consumed_sources.update(alpha_sources)
    elif alpha_sources:
        missing = sorted(_ALPHA_FACTORY_AUDIT_MEMBERS - set(alpha_sources))
        warnings.extend(f"factor_audit_missing_source:{name}" for name in missing)

    validation_sources = {
        name: available[name]
        for name in _FACTOR_VALIDATION_AUDIT_MEMBERS
        if name in available
    }
    if len(validation_sources) == len(_FACTOR_VALIDATION_AUDIT_MEMBERS):
        content, complete, detail_warnings = _build_factor_validation_audit(
            archive,
            validation_sources,
        )
        warnings.extend(detail_warnings)
        document = _derived_evidence_document(
            "derived/factor_validation_audit.json",
            content,
            validation_sources.values(),
            complete=complete,
        )
        documents.append(document)
        consumed_sources.update(validation_sources)
    elif validation_sources:
        missing = sorted(_FACTOR_VALIDATION_AUDIT_MEMBERS - set(validation_sources))
        warnings.extend(f"factor_audit_missing_source:{name}" for name in missing)

    return documents, consumed_sources, warnings


def _build_alpha_factory_candidate_audit(
    archive: zipfile.ZipFile,
    sources: dict[str, zipfile.ZipInfo],
) -> tuple[dict[str, Any], bool, list[str]]:
    candidates = _read_csv_rows(archive, sources["reports/alpha_factory_candidates.csv"])
    results = _read_csv_rows(archive, sources["reports/alpha_factory_results.csv"])
    promotions = _read_csv_rows(
        archive,
        sources["reports/alpha_factory_promotion_queue.csv"],
    )
    result_by_id, duplicate_result_ids = _rows_by_key(results, "candidate_id")
    promotion_by_id, duplicate_promotion_ids = _rows_by_key(promotions, "candidate_id")
    candidate_ids = [str(row.get("candidate_id") or "") for row in candidates]
    duplicate_candidate_ids = _duplicate_values(candidate_ids)
    candidate_id_set = {item for item in candidate_ids if item}
    result_id_set = set(result_by_id)
    promotion_id_set = set(promotion_by_id)
    rows: list[list[Any]] = []
    definition_parse_errors = 0
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        result = result_by_id.get(candidate_id, {})
        promotion = promotion_by_id.get(candidate_id, {})
        parameter_json = str(candidate.get("parameter_json") or "")
        definition_hash, parsed = _canonical_text_sha256(parameter_json)
        definition_parse_errors += int(not parsed)
        validation_metrics = _json_object(result.get("validation_metrics_json"))
        recent_metrics = _json_object(result.get("recent_7d_metrics_json"))
        cost_sources = sorted(_json_object(result.get("cost_source_mix")))
        rows.append(
            [
                candidate_id,
                str(candidate.get("template_name") or ""),
                str(candidate.get("symbol") or ""),
                str(candidate.get("regime_state") or ""),
                _compact_number(candidate.get("horizon_hours")),
                definition_hash[:16],
                _compact_number(result.get("sample_count")),
                _compact_number(result.get("avg_net_bps")),
                _compact_number(result.get("p25_net_bps")),
                _compact_number(result.get("win_rate")),
                ",".join(cost_sources),
                _compact_number(validation_metrics.get("complete_sample_count")),
                _compact_number(recent_metrics.get("complete_sample_count")),
                str(result.get("decision") or ""),
                str(promotion.get("promotion_state") or ""),
            ]
        )
    missing_results = sorted(candidate_id_set - result_id_set)
    missing_promotions = sorted(candidate_id_set - promotion_id_set)
    orphan_results = sorted(result_id_set - candidate_id_set)
    orphan_promotions = sorted(promotion_id_set - candidate_id_set)
    complete = not any(
        (
            duplicate_candidate_ids,
            duplicate_result_ids,
            duplicate_promotion_ids,
            missing_results,
            missing_promotions,
            orphan_results,
            orphan_promotions,
        )
    )
    warnings = [] if complete else ["alpha_factory_candidate_audit_join_gap"]
    content = {
        "schema_version": "quant_lab.ai_alpha_factory_candidate_audit.v1",
        "source_members": sorted(sources),
        "freshness": {
            "as_of_dates": _distinct_row_values(candidates + results + promotions, "as_of_date"),
            "generated_at_values": _bounded_distinct_row_values(
                candidates + results + promotions,
                "generated_at",
            ),
        },
        "candidate_count": len(candidates),
        "result_count": len(results),
        "promotion_count": len(promotions),
        "joined_candidate_count": len(rows),
        "join_complete": complete,
        "join_diagnostics": {
            "duplicate_candidate_ids": duplicate_candidate_ids,
            "duplicate_result_ids": duplicate_result_ids,
            "duplicate_promotion_ids": duplicate_promotion_ids,
            "missing_result_ids": missing_results,
            "missing_promotion_ids": missing_promotions,
            "orphan_result_ids": orphan_results,
            "orphan_promotion_ids": orphan_promotions,
            "definition_json_parse_error_count": definition_parse_errors,
        },
        "row_legend": [
            "candidate_id",
            "template_name",
            "symbol",
            "regime_state",
            "horizon_hours",
            "candidate_definition_sha256_prefix",
            "sample_count",
            "avg_net_bps",
            "p25_net_bps",
            "win_rate",
            "cost_sources",
            "validation_complete_sample_count",
            "recent_7d_complete_sample_count",
            "decision",
            "promotion_state",
        ],
        "rows": rows,
        "_representation": {
            "kind": "full_joined_audit",
            "selection_rule": "all_current_alpha_factory_candidates_joined_by_candidate_id",
            "row_count": len(rows),
            "truncated": False,
        },
    }
    within_budget = len(canonical_json(content)) <= FACTOR_AUDIT_MAX_DOCUMENT_CHARS
    return content, complete and within_budget, warnings


def _build_factor_validation_audit(
    archive: zipfile.ZipFile,
    sources: dict[str, zipfile.ZipInfo],
) -> tuple[dict[str, Any], bool, list[str]]:
    definitions = _read_csv_rows(archive, sources["reports/factor_definitions.csv"])
    dedupe = _read_csv_rows(archive, sources["reports/factor_dedupe_decision.csv"])
    forward = _read_csv_rows(archive, sources["reports/factor_forward_validation.csv"])
    definition_rows = [
        [
            str(row.get("factor_id") or ""),
            str(row.get("factor_family") or ""),
            str(row.get("input_features_json") or ""),
            str(row.get("template") or ""),
            str(row.get("expression_hash") or ""),
            str(row.get("canonical_factor_id") or ""),
            str(row.get("formula_hash") or ""),
            str(row.get("duplicate_of") or ""),
            str(row.get("correlation_cluster_id") or ""),
            _compact_number(row.get("independence_weight")),
            _compact_number(row.get("availability_lag_bars")),
            str(row.get("causal") or ""),
            str(row.get("operator_graph_hash") or ""),
        ]
        for row in definitions
    ]
    dedupe_rows = [
        [
            str(row.get("factor_id") or ""),
            str(row.get("correlation_cluster_id") or ""),
            _compact_number(row.get("cluster_size")),
            str(row.get("leader_factor_id") or ""),
            str(row.get("is_cluster_leader") or ""),
            _compact_number(row.get("max_abs_correlation")),
            _compact_number(row.get("independence_weight")),
            str(row.get("dedupe_decision") or ""),
            str(row.get("dedupe_reason") or ""),
        ]
        for row in dedupe
    ]
    forward_rows = [
        [
            str(row.get("factor_id") or ""),
            str(row.get("symbol") or ""),
            str(row.get("regime") or ""),
            _compact_number(row.get("horizon_hours")),
            _compact_number(row.get("sample_count")),
            _compact_number(row.get("rank_ic")),
            _compact_number(row.get("long_short_bps")),
            _compact_number(row.get("p25_net_bps")),
            _compact_number(row.get("hit_rate")),
            _compact_number(row.get("recent_7d_score")),
            _compact_number(row.get("regime_stability")),
            _compact_number(row.get("cost_adjusted_score")),
            str(row.get("recommendation") or ""),
            str(row.get("data_leakage_check") or ""),
        ]
        for row in forward
    ]
    definition_ids = {str(row.get("factor_id") or "") for row in definitions}
    forward_ids = {str(row.get("factor_id") or "") for row in forward}
    unmapped_forward_ids = sorted(item for item in forward_ids - definition_ids if item)
    complete = not unmapped_forward_ids
    warnings = [] if complete else ["factor_validation_audit_definition_gap"]
    content = {
        "schema_version": "quant_lab.ai_factor_validation_audit.v1",
        "source_members": sorted(sources),
        "freshness": {
            "definition_created_at_values": _bounded_distinct_row_values(
                definitions,
                "created_at",
            ),
            "dedupe_as_of_dates": _distinct_row_values(dedupe, "as_of_date"),
            "forward_validation_as_of_dates": _distinct_row_values(
                forward,
                "as_of_date",
            ),
        },
        "definition_count": len(definitions),
        "dedupe_decision_count": len(dedupe),
        "forward_validation_count": len(forward),
        "forward_recommendation_counts": _value_counts_rows(forward, "recommendation"),
        "unmapped_forward_factor_ids": unmapped_forward_ids,
        "definition_legend": [
            "factor_id",
            "factor_family",
            "input_features_json",
            "template",
            "expression_hash",
            "canonical_factor_id",
            "formula_hash",
            "duplicate_of",
            "correlation_cluster_id",
            "independence_weight",
            "availability_lag_bars",
            "causal",
            "operator_graph_hash",
        ],
        "definition_rows": definition_rows,
        "dedupe_legend": [
            "factor_id",
            "correlation_cluster_id",
            "cluster_size",
            "leader_factor_id",
            "is_cluster_leader",
            "max_abs_correlation",
            "independence_weight",
            "dedupe_decision",
            "dedupe_reason",
        ],
        "dedupe_rows": dedupe_rows,
        "forward_validation_legend": [
            "factor_id",
            "symbol",
            "regime",
            "horizon_hours",
            "sample_count",
            "rank_ic",
            "long_short_bps",
            "p25_net_bps",
            "hit_rate",
            "recent_7d_score",
            "regime_stability",
            "cost_adjusted_score",
            "recommendation",
            "data_leakage_check",
        ],
        "forward_validation_rows": forward_rows,
        "_representation": {
            "kind": "full_factor_validation_audit",
            "selection_rule": "all_factor_definitions_dedupe_decisions_and_forward_validation_rows",
            "truncated": False,
        },
    }
    within_budget = len(canonical_json(content)) <= FACTOR_AUDIT_MAX_DOCUMENT_CHARS
    return content, complete and within_budget, warnings


def _derived_evidence_document(
    source_member: str,
    content: dict[str, Any],
    sources: Any,
    *,
    complete: bool,
) -> EvidenceDocument:
    source_list = list(sources)
    encoded = canonical_json(content)
    return EvidenceDocument(
        source_member=source_member,
        source_format="json",
        content_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        source_size_bytes=sum(item.file_size for item in source_list),
        representation="full" if complete else "deterministic_summary",
        truncated=not complete,
        content=content,
    )


def _read_csv_rows(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
) -> list[dict[str, str]]:
    text = archive.read(member).decode("utf-8", errors="replace")
    return [
        {str(key): str(value or "") for key, value in row.items() if key is not None}
        for row in csv.DictReader(io.StringIO(text))
    ]


def _rows_by_key(
    rows: list[dict[str, str]],
    key: str,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    output: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            continue
        if value in output:
            duplicates.append(value)
            continue
        output[value] = row
    return output, sorted(set(duplicates))


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if not value:
            continue
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _canonical_text_sha256(value: str) -> tuple[str, bool]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        encoded = value
        parsed_ok = False
    else:
        encoded = canonical_json(parsed)
        parsed_ok = True
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), parsed_ok


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _compact_number(value: Any) -> int | float | str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text[:80]
    if number.is_integer():
        return int(number)
    return round(number, 6)


def _value_counts_rows(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "<empty>")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _distinct_row_values(rows: list[dict[str, str]], key: str) -> list[str]:
    return sorted({str(row.get(key) or "") for row in rows if row.get(key)})


def _bounded_distinct_row_values(
    rows: list[dict[str, str]],
    key: str,
    *,
    limit: int = 8,
) -> list[str]:
    values = _distinct_row_values(rows, key)
    if len(values) <= limit:
        return values
    return [*values[: limit // 2], *values[-(limit // 2) :]]


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


def _summarize_research_csv(
    text: str,
    *,
    max_rows: int,
    max_chars: int,
) -> tuple[dict[str, Any], bool]:
    stream = io.StringIO(text)
    try:
        reader = csv.DictReader(stream)
        fieldnames = [str(item) for item in (reader.fieldnames or [])]
        rows: list[dict[str, str]] = []
        categorical_columns = [
            column
            for column in fieldnames
            if any(
                keyword in column.lower()
                for keyword in (
                    "status",
                    "stage",
                    "decision",
                    "symbol",
                    "timeframe",
                    "regime",
                    "action",
                )
            )
        ][:16]
        category_counts: dict[str, dict[str, int]] = {
            column: {} for column in categorical_columns
        }
        row_count = 0
        selected_limit = min(max_rows, 32)
        for row in reader:
            row_count += 1
            if len(rows) < selected_limit:
                rows.append(
                    {
                        str(key): _truncate_cell_to(value, max_chars=400)
                        for key, value in row.items()
                        if key is not None
                    }
                )
            for column in categorical_columns:
                value = str(row.get(column) or "<empty>")[:160]
                counts = category_counts[column]
                if value in counts or len(counts) < 24:
                    counts[value] = counts.get(value, 0) + 1
                else:
                    counts["<other>"] = counts.get("<other>", 0) + 1
        content = {
            "columns": fieldnames,
            "row_count": row_count,
            "selected_rows": rows,
            "categorical_counts": category_counts,
            "_representation": {
                "kind": "deterministic_summary",
                "selection_rule": "first_rows_plus_bounded_categorical_counts",
                "selected_row_limit": selected_limit,
            },
        }
        while rows and len(canonical_json(content)) > max_chars:
            del rows[max(1, len(rows) // 2) :]
            content["_representation"]["selected_row_limit"] = len(rows)
        return content, len(canonical_json(content)) <= max_chars
    except csv.Error:
        return {"columns": [], "row_count": 0, "raw_prefix": text[:20_000]}, False


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
    return _truncate_cell_to(value, max_chars=1_000)


def _truncate_cell_to(value: Any, *, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


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
