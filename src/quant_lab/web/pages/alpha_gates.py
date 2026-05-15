from pathlib import Path
from typing import Any

from quant_lab.web import readers
from quant_lab.web.pages._common import lake_caption, show_frame, show_warnings, streamlit_module


def render(lake_root: str | Path, st_module: Any | None = None) -> None:
    st = streamlit_module(st_module)
    summary = readers.alpha_gate_summary(lake_root)

    st.title("Alpha Gates")
    lake_caption(st, lake_root)

    st.subheader("Alpha Discovery Board")
    for decision, count in summary["alpha_discovery_counts"].items():
        st.metric(decision, count)
    show_frame(
        st,
        summary["alpha_discovery_board"],
        "No alpha_discovery_board candidate decision rows.",
    )

    st.subheader("Strategy Evidence / Alpha Discovery Samples")
    for decision, count in summary["strategy_counts"].items():
        st.metric(decision, count)
    show_frame(
        st,
        summary["strategy_evidence"],
        "No strategy_evidence candidate discovery rows.",
    )

    st.subheader("Strategy Evidence Samples")
    show_frame(
        st,
        summary["strategy_samples"],
        "No strategy_evidence_sample rows.",
    )

    st.subheader("V5 Candidate Events")
    show_frame(
        st,
        summary["candidate_events"],
        "No v5_candidate_event rows from candidate_snapshot.csv.",
    )

    st.subheader("V5 Candidate Forward Labels")
    show_frame(
        st,
        summary["candidate_labels"],
        "No v5_candidate_label rows.",
    )

    st.subheader("V5 Candidate Outcome Summary")
    show_frame(
        st,
        summary["candidate_outcomes"],
        "No v5_candidate_outcome_summary rows.",
    )

    st.subheader("V5 Candidate Data Quality")
    show_frame(
        st,
        summary["candidate_quality"],
        "No v5_candidate_quality_daily rows.",
    )

    st.subheader("Feature Alpha Gate Decisions")
    for status, count in summary["counts"].items():
        st.metric(status, count)
    show_frame(st, summary["gates"], "No gate_decision rows.")
    show_warnings(st, summary["warnings"])
