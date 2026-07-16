#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quant_lab.ai_research.importer import import_ai_research_results
from quant_lab.ai_research.packet import (
    build_task_from_latest_export,
    build_task_from_nas_pack_reference,
    queue_status,
)
from quant_lab.export_plane.cloud_index import load_cloud_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and import read-only quant-lab AI research queue items."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-task",
        help="Build a compact AI task from the newest already-generated expert pack.",
    )
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument("--exports-dir", type=Path)
    source.add_argument("--export-queue-root", type=Path)
    build.add_argument("--queue-root", required=True, type=Path)
    build.add_argument("--force", action="store_true")
    build.add_argument("--max-member-bytes", type=int, default=256 * 1024)
    build.add_argument("--max-document-chars", type=int, default=40_000)
    build.add_argument("--max-total-chars", type=int, default=300_000)
    build.add_argument("--max-csv-rows", type=int, default=64)
    build.add_argument("--max-docs-per-section", type=int, default=4)

    import_results = subparsers.add_parser(
        "import-results",
        help="Validate NAS results and publish diagnostic-only gold datasets.",
    )
    import_results.add_argument("--queue-root", required=True, type=Path)
    import_results.add_argument("--lake-root", required=True, type=Path)
    import_results.add_argument("--max-results", type=int, default=20)

    status = subparsers.add_parser("status", help="Show queue state without reading the lake.")
    status.add_argument("--queue-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-task":
        if args.export_queue_root is not None:
            packs = load_cloud_index(args.export_queue_root)
            if not packs:
                task, task_path = None, None
            else:
                task, task_path = build_task_from_nas_pack_reference(
                    packs[0],
                    queue_root=args.queue_root,
                    force=args.force,
                )
        else:
            task, task_path = build_task_from_latest_export(
                args.exports_dir,
                queue_root=args.queue_root,
                force=args.force,
                max_member_bytes=max(64 * 1024, args.max_member_bytes),
                max_document_chars=max(10_000, args.max_document_chars),
                max_total_chars=max(100_000, args.max_total_chars),
                max_csv_rows=max(1, args.max_csv_rows),
                max_docs_per_section=max(1, args.max_docs_per_section),
            )
        payload = {
            "created": task is not None,
            "task_id": task.task_id if task else None,
            "task_path": str(task_path) if task_path else None,
            "source_pack": task.source_pack_name if task else None,
            "source_pack_id": task.source_pack_id if task else None,
            "source_location": task.source_location if task else None,
            "packet_sha256": task.packet_sha256 if task else None,
            "sections": sorted(task.sections) if task else [],
            "warnings": task.warnings if task else [],
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    if args.command == "import-results":
        payload = import_ai_research_results(
            args.queue_root,
            lake_root=args.lake_root,
            max_results=max(1, args.max_results),
        )
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if not payload["errors"] else 2

    if args.command == "status":
        print(
            json.dumps(
                queue_status(args.queue_root),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
