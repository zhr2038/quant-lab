from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_lab.export_plane.status import atomic_write_json

ATOMIC_RESEARCH_PUBLISH_SCHEMA = "quant_lab.atomic_research_publish.v1"


@dataclass(frozen=True)
class AtomicPublishItem:
    target: Path
    staged: Path


def commit_atomic_research_generation(
    lake_root: str | Path,
    *,
    transaction_name: str,
    generation_payload: dict[str, Any],
    pointer_path: Path,
    items: list[AtomicPublishItem],
    post_commit_validate: Callable[[], None] | None = None,
) -> None:
    root = Path(lake_root).resolve(strict=False)
    _require_transaction_name(transaction_name)
    recover_atomic_research_generation(
        root,
        transaction_name=transaction_name,
        pointer_path=pointer_path,
    )
    transaction_id = uuid.uuid4().hex
    journal_path = root / "gold" / f".{transaction_name}_publish_transaction.json"
    backup_root = root / "gold" / f".__{transaction_name}_backup_{transaction_id[:8]}"
    journal_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        target = _safe_path(root, item.target)
        staged = _safe_path(root, item.staged)
        if not staged.exists():
            raise FileNotFoundError(f"atomic_publish_staged_missing:{item.staged}")
        backup = backup_root / f"item-{index:03d}"
        journal_items.append(
            {
                "target": _relative(root, target),
                "staged": _relative(root, staged),
                "backup": _relative(root, backup),
                "target_existed": target.exists(),
            }
        )
    resolved_pointer = _safe_path(root, pointer_path)
    backup_root.mkdir(parents=True, exist_ok=False)
    pointer_backup = backup_root / "generation-pointer.json"
    pointer_existed = resolved_pointer.is_file()
    if pointer_existed:
        shutil.copy2(resolved_pointer, pointer_backup)
    journal = {
        "schema_version": ATOMIC_RESEARCH_PUBLISH_SCHEMA,
        "transaction_name": transaction_name,
        "transaction_id": transaction_id,
        "generation_id": generation_payload.get("generation_id"),
        "snapshot_id": generation_payload.get("snapshot_id"),
        "task_id": generation_payload.get("task_id"),
        "pointer_path": _relative(root, resolved_pointer),
        "pointer_backup": _relative(root, pointer_backup),
        "pointer_existed": pointer_existed,
        "commit_verified": False,
        "backup_root": _relative(root, backup_root),
        "items": journal_items,
    }
    atomic_write_json(journal_path, journal)
    try:
        for item in journal_items:
            target = _safe_path(root, Path(str(item["target"])))
            staged = _safe_path(root, Path(str(item["staged"])))
            backup = _safe_path(root, Path(str(item["backup"])))
            if target.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
            except Exception:
                if backup.exists() and not target.exists():
                    os.replace(backup, target)
                raise
        atomic_write_json(resolved_pointer, generation_payload)
        if post_commit_validate is not None:
            post_commit_validate()
        atomic_write_json(journal_path, journal | {"commit_verified": True})
        _complete_transaction(root, journal_path)
    except Exception:
        recover_atomic_research_generation(
            root,
            transaction_name=transaction_name,
            pointer_path=pointer_path,
        )
        raise


def recover_atomic_research_generation(
    lake_root: str | Path,
    *,
    transaction_name: str,
    pointer_path: Path,
) -> bool:
    root = Path(lake_root).resolve(strict=False)
    _require_transaction_name(transaction_name)
    journal_path = root / "gold" / f".{transaction_name}_publish_transaction.json"
    if not journal_path.is_file():
        return False
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != ATOMIC_RESEARCH_PUBLISH_SCHEMA
        or payload.get("transaction_name") != transaction_name
    ):
        raise RuntimeError("atomic_research_publish_journal_invalid")
    pointer: dict[str, Any] = {}
    resolved_pointer = _safe_path(root, pointer_path)
    if resolved_pointer.is_file():
        try:
            pointer = json.loads(resolved_pointer.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pointer = {}
    committed = bool(payload.get("commit_verified")) and all(
        pointer.get(name) == payload.get(name)
        for name in ("generation_id", "snapshot_id", "task_id")
    )
    if not committed:
        for item in reversed(list(payload.get("items") or [])):
            target = _safe_path(root, Path(str(item.get("target") or "")))
            backup = _safe_path(root, Path(str(item.get("backup") or "")))
            if backup.exists():
                _remove_path(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, target)
            elif not bool(item.get("target_existed")):
                _remove_path(target)
        pointer_backup = _safe_path(
            root,
            Path(str(payload.get("pointer_backup") or "")),
        )
        if bool(payload.get("pointer_existed")) and pointer_backup.is_file():
            resolved_pointer.parent.mkdir(parents=True, exist_ok=True)
            os.replace(pointer_backup, resolved_pointer)
        else:
            resolved_pointer.unlink(missing_ok=True)
    _complete_transaction(root, journal_path)
    return True


def _complete_transaction(root: Path, journal_path: Path) -> None:
    if not journal_path.is_file():
        return
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    backup_root = _safe_path(root, Path(str(payload.get("backup_root") or "")))
    shutil.rmtree(backup_root, ignore_errors=True)
    journal_path.unlink(missing_ok=True)


def _safe_path(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise RuntimeError("atomic_research_publish_path_escape")
    return resolved


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _require_transaction_name(value: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"
    if not value or any(character not in allowed for character in value):
        raise ValueError("unsafe_atomic_research_transaction_name")
