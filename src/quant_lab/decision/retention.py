"""Prune hot artifacts only after a signed, exact NAS archive readback acknowledgement."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from quant_lab.decision.storage import MAX_RESULT_BYTES, atomic_json, read_json
from quant_lab.export_plane.signatures import load_public_key, sha256_file, verify_payload


def prune_acknowledged(
    root: Path, public: Path, *, worker_public_key: Path, now: datetime, retention_days: int = 7
) -> dict:
    if not 7 <= retention_days <= 30:
        raise ValueError("hot retention must be between 7 and 30 days")
    ack_path = root / "inbox" / "archive-ack.json"
    if not ack_path.exists():
        return {"status": "AWAITING_NAS_ARCHIVE_ACK", "removed": []}
    ack = read_json(ack_path, max_bytes=2 * 1024**2)
    verify_payload(ack, ack.get("signature", ""), load_public_key(worker_public_key))
    if ack.get("schema_version") != "qlab.decision.archive_ack.v1":
        raise ValueError("unsupported NAS archive acknowledgement")
    at = datetime.fromisoformat(ack["generated_at"])
    if at.tzinfo is None or at > now + timedelta(seconds=5) or now - at > timedelta(days=2):
        raise ValueError("NAS archive acknowledgement is stale or future-dated")
    if len(ack["entries"]) > 10_000:
        raise ValueError("NAS archive acknowledgement exceeds entry budget")
    protected = set()
    protected_advice = set()
    for name, key in (("current-input.json", "snapshot_id"), ("current-result.json", "result_id")):
        path = root / name
        if path.exists():
            value = read_json(path, max_bytes=2 * 1024**2)
            protected.add(value[key] + ".json")
            if "input_snapshot_id" in value:
                protected.add(value["input_snapshot_id"] + ".json")
                protected_advice.update(a["advice_id"] for a in value["advice"])
    # Inputs referenced by any still-hot result/inbox must stay available for verification.
    cutoff = now - timedelta(days=retention_days)
    for folder in ("results", "inbox"):
        for path in (root / folder).glob("result-*.json"):
            value = read_json(path, max_bytes=MAX_RESULT_BYTES)
            protected.add(value["input_snapshot_id"] + ".json")
    removed = []
    # Process results first; inputs are deleted only after their evidence is acknowledged.
    for entry in sorted(ack["entries"], key=lambda item: item["kind"] == "inputs"):
        kind, name, digest = entry["kind"], entry["name"], entry["sha256"]
        prefix = {"inputs": "input", "results": "result"}.get(kind)
        if prefix is None or not re.fullmatch(prefix + r"-[a-f0-9]{64}\.json", name):
            raise ValueError("invalid acknowledgement artifact path")
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("invalid acknowledgement digest")
        path = root / kind / name
        if name in protected or not path.exists():
            continue
        if path.is_symlink() or path.resolve().parent != (root / kind).resolve():
            raise ValueError("unsafe retention path")
        if datetime.fromtimestamp(path.stat().st_mtime, UTC) >= cutoff:
            continue
        if sha256_file(path) != digest:
            raise ValueError("NAS acknowledgement does not match cloud artifact")
        raw = read_json(path, max_bytes=2 * 1024**2)
        record = {**entry, "removed_at": now.isoformat(), "ack_at": ack["generated_at"]}
        # Durable receipt before deletion allows replay after interruption.
        atomic_json(root / "retention" / name, record)
        if kind == "results":
            for advice in raw["advice"]:
                advice_id = advice["advice_id"]
                if not re.fullmatch(r"advice-[a-f0-9]{64}", advice_id):
                    raise ValueError("invalid detail identity")
                detail = public / "advice" / (advice_id + ".json")
                # Different results may reuse one immutable advice: only remove aged details.
                if (
                    advice_id not in protected_advice
                    and detail.exists()
                    and (datetime.fromtimestamp(detail.stat().st_mtime, UTC) < cutoff)
                ):
                    detail.unlink()
            (root / "receipts" / name).unlink(missing_ok=True)
        path.unlink()
        removed.append(record)
    status = {"status": "OK", "at": now.isoformat(), "removed": removed}
    atomic_json(root / "retention-status.json", status)
    return status
