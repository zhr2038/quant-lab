from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.v5_telemetry_summary(lake_root)
    latest = summary["latest"]

    st.title("V5 Telemetry")
    lake_caption(st, lake_root)
    st.metric("Status", latest.get("status", "UNKNOWN"))
    st.metric("Latest bundle ts", latest.get("latest_bundle_ts", "unknown"))
    st.metric("Latest bundle sha256", latest.get("latest_bundle_sha256", "unknown"))
    st.metric("Run count 72h", latest.get("run_count_72h", 0))
    st.metric("Decision audits 24h", latest.get("decision_audit_count_24h", 0))
    st.metric("Trade count 24h", latest.get("trade_count_24h", 0))
    st.metric("Trade count 72h", latest.get("trade_count_72h", 0))
    st.metric("Roundtrips 72h", latest.get("roundtrip_count_72h", 0))
    st.metric("Open positions", latest.get("open_position_count", 0))
    st.metric("Dust residual positions", latest.get("dust_residual_position_count", 0))
    st.metric("Kill switch", latest.get("kill_switch_enabled", "unknown"))
    st.metric("Reconcile OK", latest.get("reconcile_ok", "unknown"))
    st.metric("Ledger OK", latest.get("ledger_ok", "unknown"))
    st.metric("Auto risk level", latest.get("auto_risk_level", "unknown"))
    st.metric("High issues", latest.get("high_issue_count", 0))
    st.metric("Medium issues", latest.get("medium_issue_count", 0))
    st.metric("Config not consumed", latest.get("config_not_consumed_count", 0))
    st.metric("High score blocked", latest.get("high_score_blocked_count", 0))
    st.metric("High score matured", latest.get("high_score_blocked_matured_count", 0))

    st.subheader("Strategy Health Daily")
    show_frame(st, summary["health_rows"], "No V5 strategy health rows available.")
    st.subheader("Gate Compliance")
    show_frame(st, summary["gate_compliance_rows"], "No V5 gate compliance rows available.")
    show_warnings(st, summary["warnings"])
