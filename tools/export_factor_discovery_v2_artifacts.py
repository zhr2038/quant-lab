from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_lab.research.factor_research.registry import (  # noqa: E402
    default_hypothesis_registry,
    factor_external_audit_evidence_frame,
    factor_retirement_registry_frame,
)

MIGRATION_RECORDED_AT = datetime(2026, 7, 19, 8, 0, tzinfo=UTC)


def write_artifacts(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    retirement = factor_retirement_registry_frame(recorded_at=MIGRATION_RECORDED_AT)
    audit = factor_external_audit_evidence_frame(imported_at=MIGRATION_RECORDED_AT)
    hypotheses = default_hypothesis_registry()

    retirement.write_csv(output_root / "factor_retirement_registry.csv")
    audit.write_csv(output_root / "audit_evidence_import.csv")
    hypothesis_seed = json.dumps(
        {
            "schema_version": "factor_research_hypothesis_seed.v1",
            "generated_from": "default_hypothesis_registry",
            "recorded_at": MIGRATION_RECORDED_AT.isoformat(),
            "research_only": True,
            "live_order_effect": "none",
            "hypotheses": [item.model_dump(mode="json") for item in hypotheses],
        },
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    with (output_root / "hypothesis_registry_seed.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(hypothesis_seed + "\n")

    historical_trials = audit.select(
        "audit_version",
        "factor_id",
        "factor_family",
        "signal_validity",
        "portfolio_validity",
        "status",
        "finding",
    ).with_columns(
        (
            audit.get_column("audit_version").str.to_lowercase().str.replace_all(" ", "-")
            + "-"
            + audit.get_column("factor_id")
        ).alias("legacy_trial_reference"),
    )
    historical_trials = historical_trials.with_columns(
        trial_ledger_imported=pl.lit(False),
        not_imported_reason=pl.lit(
            "missing immutable trial identity and point-in-time snapshot"
        ),
        counts_toward_multiple_testing=pl.lit(False),
        eligible_for_promotion=pl.lit(False),
        research_only=pl.lit(True),
        live_order_effect=pl.lit("none"),
    ).select(
        "legacy_trial_reference",
        "audit_version",
        "factor_id",
        "factor_family",
        "signal_validity",
        "portfolio_validity",
        "status",
        "finding",
        "trial_ledger_imported",
        "not_imported_reason",
        "counts_toward_multiple_testing",
        "eligible_for_promotion",
        "research_only",
        "live_order_effect",
    )
    historical_trials.write_csv(output_root / "historical_trial_import.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export deterministic Factor Discovery v2 artifacts"
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()
    write_artifacts(args.output_root.resolve())


if __name__ == "__main__":
    main()
