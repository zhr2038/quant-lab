#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from quant_lab.export_plane.cloud_index import export_plane_status
from quant_lab.export_plane.receipt import import_export_receipts
from quant_lab.export_plane.request import process_export_requests, submit_export_request


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the NAS-local Expert Export queue.")
    commands = parser.add_subparsers(dest="command", required=True)
    request = commands.add_parser("request")
    request.add_argument("--queue-root", required=True, type=Path)
    request.add_argument("--date", required=True, type=date.fromisoformat)
    request.add_argument("--mode", choices=("cached", "authoritative"), default="authoritative")

    process = commands.add_parser("process-requests")
    process.add_argument("--queue-root", required=True, type=Path)
    process.add_argument("--lake-root", required=True, type=Path)
    process.add_argument("--private-key", required=True, type=Path)
    process.add_argument("--key-id", required=True)
    process.add_argument("--worker-commit", required=True)

    receipts = commands.add_parser("import-receipts")
    receipts.add_argument("--queue-root", required=True, type=Path)
    receipts.add_argument("--worker-public-key", required=True, type=Path)
    receipts.add_argument("--worker-key-id", required=True)

    status = commands.add_parser("status")
    status.add_argument("--queue-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "request":
        request, created = submit_export_request(
            queue_root=args.queue_root,
            export_date=args.date,
            export_mode=args.mode,
            requested_by=os.getenv("USER", "operator"),
        )
        payload = {**request.model_dump(mode="json"), "created": created}
    elif args.command == "process-requests":
        payload = process_export_requests(
            queue_root=args.queue_root,
            lake_root=args.lake_root,
            signing_key_path=args.private_key,
            signature_key_id=args.key_id,
            expected_worker_commit=args.worker_commit.lower(),
        )
    elif args.command == "import-receipts":
        payload = import_export_receipts(
            queue_root=args.queue_root,
            worker_public_key_path=args.worker_public_key,
            worker_key_id=args.worker_key_id,
        )
    elif args.command == "status":
        payload = export_plane_status(args.queue_root)
    else:
        raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
