# ruff: noqa: E501
"""Generate read-only empirical cost scenarios and evidence limitations."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.costs import extract_real_cost_evidence  # noqa: E402
from audit.auditlib.paths import ARTIFACTS, REPORTS  # noqa: E402


def main() -> None:
    summary, metadata = extract_real_cost_evidence()
    summary.write_csv(ARTIFACTS / "cost_model_summary.csv")
    (ARTIFACTS / "cost_model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    lines = [
        "# Cost Model Report",
        "",
        f"- generated_at: {datetime.now(UTC).isoformat()}",
        f"- V5 fill rows: {metadata['fill_fee']['fill_rows']}; convertible fee rows: {metadata['fill_fee']['fee_converted_rows']}.",
        f"- Actual qlab cost-bucket rows: {metadata['actual_bucket_rows']}; symbols: {', '.join(metadata['actual_symbols'])}.",
        "- Scenario mapping is fixed: optimistic=p50, base=p75, stress=p90; one-way cost never falls below the production 10bps fee + 5bps slippage fallback.",
        "- The backtests deduct two one-way legs per unit of portfolio turnover.",
        "- Cost coverage outside four V5 symbols is insufficient; this blocks broad-universe PASS even when scenario returns are positive.",
        f"- Limitation: {metadata['limitation']}",
        "",
        "| Scenario | One-way bps | Roundtrip bps | Empirical cost | Fee | Slippage | Spread | Evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f"| {row['scenario']} | {row['one_way_cost_bps']:.3f} | {row['roundtrip_cost_bps']:.3f} | "
            f"{row['empirical_total_cost_bps']} | {row['fee_bps']} | {row['empirical_slippage_bps']} | "
            f"{row['empirical_spread_bps']} | {row['evidence_status']} |"
        )
    (REPORTS / "cost_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
