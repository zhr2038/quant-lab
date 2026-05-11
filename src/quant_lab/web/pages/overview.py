from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    snapshot = readers.dashboard_overview(lake_root)

    st.title("quant-lab Overview")
    lake_caption(st, lake_root)
    st.caption("Read-only observability for OKX-first research data and strategy permissions.")

    diagnostics = snapshot["diagnostics"]
    st.subheader("Diagnostics")
    st.metric("lake_root", diagnostics["lake_root"])
    st.metric("lake_root_exists", str(diagnostics["lake_root_exists"]))
    st.metric("lake_root_parquet_file_count", diagnostics["parquet_file_count"])
    st.metric(
        "latest_market_bar_ts",
        str(diagnostics["latest_market_bar_ts"] or "unknown"),
    )
    show_frame(
        st,
        diagnostics["datasets"],
        "No core dataset diagnostics available.",
    )
    show_warnings(st, diagnostics["warnings"])

    st.metric("quant-lab status", snapshot["status"])
    st.metric("V5 permission", snapshot["v5_permission"])
    st.metric("V7 permission", snapshot["v7_permission"])
    st.metric("Latest market bar", str(snapshot["latest_market_bar_ts"] or "unknown"))
    st.metric("Missing bar ratio", f"{snapshot['missing_bar_ratio']:.2%}")
    st.metric("Cost fallback ratio", f"{snapshot['cost_fallback_ratio']:.2%}")

    collector_rows = [
        {"collector": "OKX public REST", "status": snapshot["okx_public_rest_status"]},
        {"collector": "OKX public WebSocket", "status": snapshot["okx_public_ws_status"]},
        {"collector": "OKX read-only private", "status": snapshot["okx_readonly_status"]},
    ]
    st.subheader("Collector Status")
    show_frame(st, pl.DataFrame(collector_rows), "No collector status available.")

    st.subheader("Alpha Gate Counts")
    gate_counts = pl.DataFrame(
        [
            {"status": status, "count": count}
            for status, count in snapshot["alpha_gate_counts"].items()
        ]
    )
    show_frame(st, gate_counts, "No gate decisions available.")

    st.subheader("Latest Expert Pack")
    st.write(snapshot["latest_expert_pack"] or "No expert pack found.")
    diagnostic_warnings = set(diagnostics["warnings"])
    show_warnings(
        st,
        [warning for warning in snapshot["warnings"] if warning not in diagnostic_warnings],
    )
