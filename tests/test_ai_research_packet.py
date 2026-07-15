from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from datetime import UTC, datetime

from quant_lab.ai_research.contracts import canonical_json, compute_task_packet_sha256
from quant_lab.ai_research.packet import (
    _summarize_data_quality,
    build_ai_research_task,
    find_latest_expert_pack,
    queue_status,
)


def test_packet_uses_existing_expert_pack_and_deduplicates(tmp_path) -> None:
    pack = tmp_path / "quant_lab_expert_pack_2026-07-14.zip"
    full_csv = "factor_id,rank_ic_mean,long_short_mean_bps\n" + "\n".join(
        f"factor-{index},{index / 1000:.4f},{index / 10:.2f}" for index in range(500)
    )
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps({"fresh": True, "pack": "test"}))
        archive.writestr("data_quality.json", json.dumps({"status": "PASS"}))
        archive.writestr("reports/factor_evidence.csv", full_csv)
        archive.writestr(
            "reports/v5_trade_outcome_attribution.csv",
            "category,count\nentry_bad,2\nexit_bad,7\n",
        )
        archive.writestr("restricted/secret.txt", "must never be selected")

    queue = tmp_path / "queue"
    created_at = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)
    task, task_path = build_ai_research_task(
        pack,
        queue_root=queue,
        max_member_bytes=256,
        max_document_chars=10_000,
        max_total_chars=100_000,
        max_csv_rows=3,
        max_docs_per_section=4,
        now=created_at,
    )

    assert task is not None
    assert task_path is not None and task_path.is_file()
    assert compute_task_packet_sha256(task) == task.packet_sha256
    assert "core_state" in task.sections
    assert "factor_research" in task.sections
    assert "trade_learning" in task.sections
    assert task.preflight is not None
    assert task.preflight.status == "BLOCK"
    assert task.preflight.missing_core_members == ["provenance.json"]
    assert all(
        "restricted" not in doc.source_member
        for docs in task.sections.values()
        for doc in docs
    )

    factor_doc = next(
        doc
        for doc in task.sections["factor_research"]
        if doc.source_member.endswith("factor_evidence.csv")
    )
    assert factor_doc.truncated is True
    assert factor_doc.content_sha256 == hashlib.sha256(
        full_csv.encode("utf-8")[:256]
    ).hexdigest()
    assert factor_doc.source_size_bytes == len(full_csv.encode("utf-8"))
    assert len(factor_doc.content["rows"]) == 3

    duplicate, duplicate_path = build_ai_research_task(
        pack,
        queue_root=queue,
        max_member_bytes=256,
        max_document_chars=10_000,
        max_total_chars=100_000,
        max_csv_rows=3,
        max_docs_per_section=4,
        now=created_at,
    )
    assert duplicate is None
    assert duplicate_path is None
    assert queue_status(queue)["counts"]["pending"] == 1
    assert (task_path.parent / "task_manifest.json").is_file()
    assert not list((queue / ".staging").iterdir())
    if os.name != "nt":
        assert stat.S_IMODE(task_path.parent.stat().st_mode) == 0o2770
        assert stat.S_IMODE(task_path.stat().st_mode) == 0o660
        assert stat.S_IMODE((task_path.parent / "task_manifest.json").stat().st_mode) == 0o660


def test_latest_expert_pack_skips_newer_partial_zip(tmp_path) -> None:
    valid = tmp_path / "quant_lab_expert_pack_2026-07-14_valid.zip"
    with zipfile.ZipFile(valid, "w") as archive:
        archive.writestr("manifest.json", "{}")
    partial = tmp_path / "quant_lab_expert_pack_2026-07-14_partial.zip"
    partial.write_bytes(b"PK\x03\x04unfinished")
    valid.touch()
    partial.touch()

    assert find_latest_expert_pack(tmp_path) == valid


def test_default_packet_stays_within_model_input_budget(tmp_path) -> None:
    pack = tmp_path / "quant_lab_expert_pack_2026-07-14_large.zip"
    large_csv = "factor_id,rank_ic_mean\n" + "\n".join(
        f"factor-{index},{index / 1000:.4f}" for index in range(20_000)
    )
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps({"fresh": True}))
        for index in range(8):
            archive.writestr(f"reports/factor_evidence_{index}.csv", large_csv)

    task, _ = build_ai_research_task(pack, queue_root=tmp_path / "queue")

    assert task is not None
    assert len(canonical_json(task.model_dump(mode="json")).encode("utf-8")) < 400_000


def test_packet_preflight_passes_with_complete_core_identity(tmp_path) -> None:
    pack = tmp_path / "quant_lab_expert_pack_complete.zip"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("provenance.json", "{}")
        archive.writestr("data_quality.json", "{}")

    task, _ = build_ai_research_task(pack, queue_root=tmp_path / "queue")

    assert task is not None and task.preflight is not None
    assert task.preflight.status == "PASS"
    assert task.preflight.blockers == []


def test_packet_preflight_warns_but_does_not_block_truncated_core(tmp_path) -> None:
    pack = tmp_path / "quant_lab_expert_pack_large_core.zip"
    large = json.dumps({"rows": [{"value": "x" * (2 * 1024 * 1024 + 1_024)}]})
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", large)
        archive.writestr("provenance.json", "{}")
        archive.writestr("data_quality.json", "{}")

    task, _ = build_ai_research_task(
        pack,
        queue_root=tmp_path / "queue",
        max_document_chars=500,
    )

    assert task is not None and task.preflight is not None
    assert task.preflight.status == "WARN"
    assert task.preflight.blockers == []
    assert any(item.startswith("truncated_core_member:") for item in task.preflight.warnings)


def test_large_core_json_uses_complete_deterministic_summary(tmp_path) -> None:
    pack = tmp_path / "quant_lab_expert_pack_large_core_summary.zip"
    manifest = {
        "generated_at": "2026-07-14T00:00:00Z",
        "files": [{"path": f"reports/{index}.csv"} for index in range(2_000)],
        "row_counts": {f"factor_{index}": index for index in range(200)},
        "dataset_freshness": {
            f"factor_{index}": {"status": "OK", "latest_ts": "2026-07-14T00:00:00Z"}
            for index in range(200)
        },
    }
    data_quality = {
        "status": "WARN",
        "checks": [
            {"dataset": f"factor_{index}", "status": "PASS", "detail": "x" * 500}
            for index in range(500)
        ],
        "warnings": ["bounded warning"],
        "dataset_governance": {
            "status": "WARN",
            "checks": [
                {"dataset": f"factor_{index}", "status": "PASS", "detail": "y" * 500}
                for index in range(500)
            ],
        },
    }
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("provenance.json", "{}")
        archive.writestr("data_quality.json", json.dumps(data_quality))

    task, _ = build_ai_research_task(
        pack,
        queue_root=tmp_path / "queue",
        max_member_bytes=256,
        max_document_chars=40_000,
    )

    assert task is not None and task.preflight is not None
    core = {item.source_member: item for item in task.sections["core_state"]}
    assert task.preflight.status == "PASS"
    assert core["manifest.json"].representation == "deterministic_summary"
    assert core["data_quality.json"].representation == "deterministic_summary"
    assert core["manifest.json"].truncated is False
    assert core["data_quality.json"].truncated is False
    assert core["manifest.json"].content["_representation"]["file_count"] == 2_000
    assert core["data_quality.json"].content["checks"]["count"] == 500


def test_large_alpha_board_uses_complete_deterministic_summary(tmp_path) -> None:
    pack = tmp_path / "quant_lab_expert_pack_large_alpha_board.zip"
    board = "factor_id,symbol,timeframe,status,score,detail\n" + "\n".join(
        f"factor-{index},SOL-USDT,8h,SHADOW,{index},{'x' * 300}"
        for index in range(5_000)
    )
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("provenance.json", "{}")
        archive.writestr("data_quality.json", "{}")
        archive.writestr("reports/alpha_discovery_board.csv", board)

    task, _ = build_ai_research_task(
        pack,
        queue_root=tmp_path / "queue",
        max_member_bytes=256,
        max_document_chars=40_000,
    )

    assert task is not None and task.preflight is not None
    board_doc = next(
        item
        for item in task.sections["factor_research"]
        if item.source_member == "reports/alpha_discovery_board.csv"
    )
    assert task.preflight.status == "PASS"
    assert board_doc.representation == "deterministic_summary"
    assert board_doc.truncated is False
    assert board_doc.content["row_count"] == 5_000
    assert board_doc.content["categorical_counts"]["status"] == {"SHADOW": 5_000}
    assert len(canonical_json(board_doc.content)) <= 40_000


def test_factor_audit_documents_preserve_complete_candidate_and_validation_rows(
    tmp_path,
) -> None:
    pack = tmp_path / "quant_lab_expert_pack_factor_audit.zip"
    candidate_header = (
        "candidate_id,template_name,symbol,regime_state,horizon_hours,parameter_json\n"
    )
    result_header = (
        "candidate_id,sample_count,avg_net_bps,p25_net_bps,win_rate,"
        "cost_source_mix,validation_metrics_json,recent_7d_metrics_json,decision\n"
    )
    promotion_header = "candidate_id,promotion_state\n"
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("provenance.json", "{}")
        archive.writestr("data_quality.json", "{}")
        archive.writestr(
            "reports/alpha_factory_candidates.csv",
            candidate_header
            + 'candidate-a,feature,SOL-USDT,TREND_UP,8,"{""feature"":""f1""}"\n'
            + 'candidate-b,product,ETH-USDT,RISK_OFF,24,"{""left"":""f1""}"\n',
        )
        archive.writestr(
            "reports/alpha_factory_results.csv",
            result_header
            + 'candidate-a,40,12.5,-2.0,0.6,"{""actual"":40}",'
            '"{""complete_sample_count"":20}",'
            '"{""complete_sample_count"":8}",KEEP_SHADOW\n'
            + 'candidate-b,50,8.5,-4.0,0.55,"{""proxy"":50}",'
            '"{""complete_sample_count"":22}",'
            '"{""complete_sample_count"":9}",RESEARCH\n',
        )
        archive.writestr(
            "reports/alpha_factory_promotion_queue.csv",
            promotion_header + "candidate-a,KEEP_SHADOW\ncandidate-b,RESEARCH\n",
        )
        archive.writestr(
            "reports/factor_definitions.csv",
            "factor_id,factor_family,input_features_json,template,expression_hash,"
            "canonical_factor_id,formula_hash,duplicate_of,correlation_cluster_id,"
            "independence_weight,availability_lag_bars,causal,operator_graph_hash\n"
            'f1,momentum,"[""close_return_24""]",feature,expr-1,canonical-1,'
            "formula-1,,cluster-1,1.0,1,True,graph-1\n",
        )
        archive.writestr(
            "reports/factor_dedupe_decision.csv",
            "factor_id,correlation_cluster_id,cluster_size,leader_factor_id,"
            "is_cluster_leader,max_abs_correlation,independence_weight,"
            "dedupe_decision,dedupe_reason\n"
            "f1,cluster-1,1,f1,True,1.0,1.0,keep_leader,unique\n",
        )
        archive.writestr(
            "reports/factor_forward_validation.csv",
            "factor_id,symbol,regime,horizon_hours,sample_count,rank_ic,"
            "long_short_bps,p25_net_bps,hit_rate,recent_7d_score,regime_stability,"
            "cost_adjusted_score,recommendation,data_leakage_check\n"
            "f1,SOL-USDT,TREND_UP,8,40,0.04,12.5,-2.0,0.6,5.0,0.8,12.5,"
            "FORWARD_VALIDATION_PASS,pass\n",
        )

    task, _ = build_ai_research_task(pack, queue_root=tmp_path / "queue")

    assert task is not None and task.preflight is not None
    documents = {item.source_member: item for item in task.sections["factor_research"]}
    alpha = documents["derived/alpha_factory_candidate_audit.json"]
    factor = documents["derived/factor_validation_audit.json"]
    assert alpha.truncated is False
    assert alpha.content["join_complete"] is True
    assert alpha.content["candidate_count"] == 2
    assert alpha.content["joined_candidate_count"] == 2
    assert [row[0] for row in alpha.content["rows"]] == ["candidate-a", "candidate-b"]
    assert alpha.content["rows"][0][11] == 20
    assert alpha.content["rows"][0][12] == 8
    assert factor.truncated is False
    assert factor.content["definition_count"] == 1
    assert factor.content["forward_validation_count"] == 1
    assert factor.content["forward_validation_rows"][0][0] == "f1"
    assert task.preflight.status == "PASS"


def test_data_quality_summary_does_not_treat_pass_severity_as_failure() -> None:
    summary = _summarize_data_quality(
        {
            "checks": [
                {"name": "critical_when_failed", "status": "PASS", "severity": "critical"},
                {"name": "real_warning", "status": "WARNING", "severity": "warning"},
            ]
        }
    )

    assert summary["checks"]["count"] == 2
    assert summary["checks"]["non_ok_count"] == 1
    assert summary["checks"]["selected"][0]["name"] == "real_warning"
