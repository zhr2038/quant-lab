from __future__ import annotations

from collections import defaultdict
from typing import Any

import polars as pl

from quant_lab.strategy_telemetry.sanitize import safe_json_dumps

CORRELATION_CLUSTER_THRESHOLD = 0.90
FACTOR_FACTORY_V2_LIVE_ORDER_EFFECT = "none_read_only_research"
FAST_MICROSTRUCTURE_FORWARD_AGGREGATE_REGIME = "ALL_REGIMES"
STRATEGY_REVIEW_BLOCKING_REASONS = [
    "needs_strategy_formulation",
    "needs_paper_tracking",
    "needs_cost_validation",
]

FACTOR_DEDUPE_DECISION_FIELDS = [
    "as_of_date",
    "factor_id",
    "factor_family",
    "candidate_state",
    "correlation_cluster_id",
    "cluster_size",
    "is_cluster_leader",
    "leader_factor_id",
    "max_abs_correlation",
    "dedupe_decision",
    "dedupe_reason",
    "live_order_effect",
]

FACTOR_FAMILY_LEADERBOARD_FIELDS = [
    "as_of_date",
    "factor_family",
    "factor_count",
    "paper_ready_count",
    "leader_factor_id",
    "leader_candidate_state",
    "leader_best_horizon_bars",
    "leader_best_rank_ic_tstat",
    "leader_best_long_short_mean_bps",
    "high_correlation_cluster_count",
    "recommendation",
    "live_order_effect",
]

FACTOR_PAPER_REVIEW_QUEUE_FIELDS = [
    "as_of_date",
    "factor_id",
    "factor_family",
    "candidate_state",
    "best_horizon_bars",
    "best_rank_ic_mean",
    "best_rank_ic_tstat",
    "best_long_short_mean_bps",
    "sample_count",
    "oos_score",
    "regime_stability_score",
    "correlation_cluster_id",
    "recommendation",
    "live_order_effect",
]

COMPOSITE_FACTOR_CANDIDATE_FIELDS = [
    "as_of_date",
    "composite_factor_id",
    "factor_terms",
    "available_term_count",
    "missing_terms",
    "max_terms",
    "interpretable_only",
    "no_future_fields",
    "leakage_check",
    "component_candidate_states",
    "recommendation",
    "live_order_effect",
]

FACTOR_REGIME_EFFECTIVENESS_FIELDS = [
    "as_of_date",
    "factor_id",
    "regime",
    "horizon",
    "rank_ic",
    "long_short_bps",
    "win_rate",
    "sample_count",
    "recommendation",
    "live_order_effect",
]

FACTOR_STRATEGY_BRIDGE_CANDIDATE_FIELDS = [
    "as_of_date",
    "factor_id",
    "factor_family",
    "correlation_cluster_id",
    "symbol",
    "regime",
    "horizon",
    "horizon_hours",
    "forward_sample_count",
    "forward_cost_adjusted_score",
    "bridge_candidate_id",
    "eligible_for_alpha_factory",
    "blocking_reasons",
    "recommended_action",
    "live_order_effect",
]

COMPOSITE_FACTOR_RECIPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "composite.liquidity_momentum_volume_24",
        ("core.liquidity_adjusted_momentum_24", "core.volume_zscore_24"),
    ),
    (
        "composite.close_return_position_24",
        ("core.close_return_24", "core.close_position_in_range"),
    ),
    (
        "composite.momentum_market_pressure",
        ("core.momentum_vol_adjusted_24", "market_pressure_state"),
    ),
    (
        "composite.bottom_zone_sell_exhaustion",
        ("bottom_zone_score", "taker_sell_exhaustion"),
    ),
    (
        "composite.late_breakout_vwap_failure",
        ("late_breakout_failure_score", "distance_to_vwap"),
    ),
    (
        "composite.volume_spread_normalization",
        ("core.volume_zscore_24", "spread_normalization"),
    ),
)

REGIME_BUCKETS = (
    "RISK_OFF",
    "SIDEWAYS",
    "TREND_UP",
    "TREND_DOWN",
    "ALT_IMPULSE",
    "LOW_VOL",
)


def build_factor_factory_v2_reports(
    *,
    candidates: pl.DataFrame,
    evidence: pl.DataFrame,
    correlations: pl.DataFrame,
    factor_forward_validation: pl.DataFrame | None = None,
    fast_microstructure_forward_test: pl.DataFrame | None = None,
) -> dict[str, pl.DataFrame]:
    dedupe = build_factor_dedupe_decision(candidates=candidates, correlations=correlations)
    leaderboard = build_factor_family_leaderboard(candidates=candidates, dedupe=dedupe)
    paper_queue = build_factor_paper_review_queue(
        candidates=candidates,
        evidence=evidence,
        dedupe=dedupe,
    )
    composite = build_composite_factor_candidates(candidates=candidates)
    regime = build_factor_regime_effectiveness(candidates=candidates, evidence=evidence)
    bridge = build_factor_strategy_bridge_candidates(
        paper_queue=paper_queue,
        factor_forward_validation=factor_forward_validation,
        fast_microstructure_forward_test=fast_microstructure_forward_test,
    )
    return {
        "factor_dedupe_decision": dedupe,
        "factor_family_leaderboard": leaderboard,
        "factor_paper_review_queue": paper_queue,
        "composite_factor_candidates": composite,
        "factor_regime_effectiveness": regime,
        "factor_strategy_bridge_candidates": bridge,
    }


def build_factor_dedupe_decision(
    *,
    candidates: pl.DataFrame,
    correlations: pl.DataFrame,
    threshold: float = CORRELATION_CLUSTER_THRESHOLD,
) -> pl.DataFrame:
    candidate_rows = _rows(candidates)
    if not candidate_rows:
        return _frame([], FACTOR_DEDUPE_DECISION_FIELDS)
    by_factor = {
        str(row.get("factor_id") or ""): row
        for row in candidate_rows
        if row.get("factor_id")
    }
    factor_ids = sorted(by_factor)
    parent = {factor_id: factor_id for factor_id in factor_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    max_corr: dict[str, float] = {factor_id: 0.0 for factor_id in factor_ids}
    for row in _rows(correlations):
        left = str(row.get("factor_id_left") or "")
        right = str(row.get("factor_id_right") or "")
        corr = abs(_float(row.get("correlation")) or 0.0)
        if left not in by_factor or right not in by_factor:
            continue
        max_corr[left] = max(max_corr.get(left, 0.0), corr)
        max_corr[right] = max(max_corr.get(right, 0.0), corr)
        if corr >= threshold:
            union(left, right)

    clusters: dict[str, list[str]] = defaultdict(list)
    for factor_id in factor_ids:
        clusters[find(factor_id)].append(factor_id)
    ordered_clusters = sorted(clusters.values(), key=lambda values: sorted(values)[0])
    cluster_ids = {
        factor_id: f"cluster_{index:03d}"
        for index, values in enumerate(ordered_clusters, start=1)
        for factor_id in sorted(values)
    }
    leaders: dict[str, str] = {}
    for values in ordered_clusters:
        leader = sorted(
            values,
            key=lambda factor_id: _candidate_rank_key(by_factor[factor_id]),
            reverse=True,
        )[0]
        for factor_id in values:
            leaders[factor_id] = leader

    out = []
    for factor_id in factor_ids:
        row = by_factor[factor_id]
        leader = leaders[factor_id]
        cluster = clusters[find(factor_id)]
        is_leader = factor_id == leader
        out.append(
            {
                "as_of_date": row.get("as_of_date"),
                "factor_id": factor_id,
                "factor_family": row.get("factor_family"),
                "candidate_state": row.get("candidate_state"),
                "correlation_cluster_id": cluster_ids[factor_id],
                "cluster_size": len(cluster),
                "is_cluster_leader": is_leader,
                "leader_factor_id": leader,
                "max_abs_correlation": round(max_corr.get(factor_id, 0.0), 6),
                "dedupe_decision": "keep_leader" if is_leader else "redundant_suppressed",
                "dedupe_reason": (
                    "cluster_leader_selected_by_state_tstat_spread"
                    if is_leader
                    else f"correlation_ge_{threshold:.2f}_leader={leader}"
                ),
                "live_order_effect": FACTOR_FACTORY_V2_LIVE_ORDER_EFFECT,
            }
        )
    return _frame(out, FACTOR_DEDUPE_DECISION_FIELDS)


def build_factor_family_leaderboard(
    *,
    candidates: pl.DataFrame,
    dedupe: pl.DataFrame,
) -> pl.DataFrame:
    candidate_rows = _rows(candidates)
    if not candidate_rows:
        return _frame([], FACTOR_FAMILY_LEADERBOARD_FIELDS)
    dedupe_by_factor = {row.get("factor_id"): row for row in _rows(dedupe)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row.get("factor_family") or "unknown")].append(row)
    out = []
    for family, values in sorted(grouped.items()):
        leaders = [
            row
            for row in values
            if bool((dedupe_by_factor.get(row.get("factor_id")) or {}).get("is_cluster_leader"))
        ] or values
        leader = sorted(leaders, key=_candidate_rank_key, reverse=True)[0]
        cluster_ids = {
            (dedupe_by_factor.get(row.get("factor_id")) or {}).get("correlation_cluster_id")
            for row in values
            if int((dedupe_by_factor.get(row.get("factor_id")) or {}).get("cluster_size") or 1) > 1
        }
        out.append(
            {
                "as_of_date": leader.get("as_of_date"),
                "factor_family": family,
                "factor_count": len(values),
                "paper_ready_count": sum(
                    1 for row in values if str(row.get("candidate_state") or "") == "PAPER_READY"
                ),
                "leader_factor_id": leader.get("factor_id"),
                "leader_candidate_state": leader.get("candidate_state"),
                "leader_best_horizon_bars": leader.get("best_horizon_bars"),
                "leader_best_rank_ic_tstat": leader.get("best_rank_ic_tstat"),
                "leader_best_long_short_mean_bps": leader.get("best_long_short_mean_bps"),
                "high_correlation_cluster_count": len({item for item in cluster_ids if item}),
                "recommendation": (
                    "REVIEW_FAMILY_LEADER_FOR_PAPER"
                    if str(leader.get("candidate_state") or "") == "PAPER_READY"
                    else "KEEP_RESEARCH"
                ),
                "live_order_effect": FACTOR_FACTORY_V2_LIVE_ORDER_EFFECT,
            }
        )
    return _frame(out, FACTOR_FAMILY_LEADERBOARD_FIELDS)


def build_factor_paper_review_queue(
    *,
    candidates: pl.DataFrame,
    evidence: pl.DataFrame,
    dedupe: pl.DataFrame,
) -> pl.DataFrame:
    dedupe_by_factor = {row.get("factor_id"): row for row in _rows(dedupe)}
    evidence_by_factor = _best_evidence_by_factor(evidence)
    out = []
    for row in _rows(candidates):
        if str(row.get("candidate_state") or "") != "PAPER_READY":
            continue
        factor_id = str(row.get("factor_id") or "")
        dedupe_row = dedupe_by_factor.get(factor_id, {})
        evidence_row = evidence_by_factor.get(factor_id, {})
        suppressed = str(dedupe_row.get("dedupe_decision") or "") == "redundant_suppressed"
        oos_score = _oos_score(row, evidence_row)
        regime_score = _regime_stability_score(factor_id, evidence)
        out.append(
            {
                "as_of_date": row.get("as_of_date"),
                "factor_id": factor_id,
                "factor_family": row.get("factor_family"),
                "candidate_state": row.get("candidate_state"),
                "best_horizon_bars": row.get("best_horizon_bars"),
                "best_rank_ic_mean": row.get("best_rank_ic_mean"),
                "best_rank_ic_tstat": row.get("best_rank_ic_tstat"),
                "best_long_short_mean_bps": row.get("best_long_short_mean_bps"),
                "sample_count": evidence_row.get("valid_sample_count")
                or evidence_row.get("sample_count"),
                "oos_score": oos_score,
                "regime_stability_score": regime_score,
                "correlation_cluster_id": dedupe_row.get("correlation_cluster_id"),
                "recommendation": "HOLD_REVIEW_REDUNDANT" if suppressed else "FACTOR_PAPER_REVIEW",
                "live_order_effect": FACTOR_FACTORY_V2_LIVE_ORDER_EFFECT,
            }
        )
    return _frame(out, FACTOR_PAPER_REVIEW_QUEUE_FIELDS)


def build_composite_factor_candidates(*, candidates: pl.DataFrame) -> pl.DataFrame:
    candidate_by_id = {str(row.get("factor_id") or ""): row for row in _rows(candidates)}
    out = []
    as_of_date = _latest_as_of_date(_rows(candidates))
    for composite_id, terms in COMPOSITE_FACTOR_RECIPES:
        available = [
            term
            for term in terms
            if term in candidate_by_id or _external_research_term(term)
        ]
        missing = [term for term in terms if term not in available]
        component_states = {
            term: str(
                (candidate_by_id.get(term) or {}).get("candidate_state")
                or "EXTERNAL_RESEARCH"
            )
            for term in available
        }
        out.append(
            {
                "as_of_date": as_of_date,
                "composite_factor_id": composite_id,
                "factor_terms": safe_json_dumps(list(terms)),
                "available_term_count": len(available),
                "missing_terms": safe_json_dumps(missing),
                "max_terms": 3,
                "interpretable_only": True,
                "no_future_fields": True,
                "leakage_check": "pass_no_future_label_fields",
                "component_candidate_states": safe_json_dumps(component_states),
                "recommendation": (
                    "COMPOSITE_RESEARCH_QUEUE"
                    if not missing and len(terms) <= 3
                    else "WAIT_FOR_COMPONENT_FACTORS"
                ),
                "live_order_effect": FACTOR_FACTORY_V2_LIVE_ORDER_EFFECT,
            }
        )
    return _frame(out, COMPOSITE_FACTOR_CANDIDATE_FIELDS)


def build_factor_regime_effectiveness(
    *,
    candidates: pl.DataFrame,
    evidence: pl.DataFrame,
) -> pl.DataFrame:
    evidence_rows = _rows(evidence)
    regime_column = _first_existing_column(evidence, ("regime", "regime_state", "market_regime"))
    out = []
    if regime_column:
        for row in evidence_rows:
            out.append(
                {
                    "as_of_date": row.get("as_of_date"),
                    "factor_id": row.get("factor_id"),
                    "regime": row.get(regime_column),
                    "horizon": row.get("horizon_bars"),
                    "rank_ic": row.get("rank_ic_mean"),
                    "long_short_bps": row.get("long_short_mean_bps"),
                    "win_rate": row.get("win_rate"),
                    "sample_count": row.get("valid_sample_count") or row.get("sample_count"),
                    "recommendation": _regime_recommendation(row),
                    "live_order_effect": FACTOR_FACTORY_V2_LIVE_ORDER_EFFECT,
                }
            )
        return _frame(out, FACTOR_REGIME_EFFECTIVENESS_FIELDS)

    best_evidence = _best_evidence_by_factor(evidence)
    for row in _rows(candidates):
        factor_id = str(row.get("factor_id") or "")
        evidence_row = best_evidence.get(factor_id, {})
        for regime in REGIME_BUCKETS:
            out.append(
                {
                    "as_of_date": row.get("as_of_date"),
                    "factor_id": factor_id,
                    "regime": regime,
                    "horizon": row.get("best_horizon_bars"),
                    "rank_ic": None,
                    "long_short_bps": None,
                    "win_rate": None,
                    "sample_count": 0,
                    "recommendation": (
                        "NEEDS_REGIME_LABELED_EVIDENCE"
                        if evidence_row
                        else "NEEDS_FACTOR_EVIDENCE"
                    ),
                    "live_order_effect": FACTOR_FACTORY_V2_LIVE_ORDER_EFFECT,
                }
            )
    return _frame(out, FACTOR_REGIME_EFFECTIVENESS_FIELDS)


def build_factor_strategy_bridge_candidates(
    *,
    paper_queue: pl.DataFrame,
    factor_forward_validation: pl.DataFrame | None = None,
    fast_microstructure_forward_test: pl.DataFrame | None = None,
) -> pl.DataFrame:
    forward_pass_by_factor = _forward_validation_pass_by_factor(factor_forward_validation)
    forward_pass_rows_by_factor = _forward_validation_pass_rows_by_factor(
        factor_forward_validation
    )
    forward_context_by_factor = _forward_validation_pass_context_by_factor(
        factor_forward_validation
    )
    seen_factor_ids: set[str] = set()
    out = []
    for row in _rows(paper_queue):
        reasons = []
        factor_id = str(row.get("factor_id") or "")
        seen_factor_ids.add(factor_id)
        forward_passed = forward_pass_by_factor.get(factor_id, False)
        paper_review = str(row.get("recommendation") or "") == "FACTOR_PAPER_REVIEW"
        if not paper_review:
            reasons.append("factor_redundant_or_not_paper_review")
        if (_float(row.get("oos_score")) or 0.0) <= 0:
            reasons.append("oos_not_positive_or_missing")
        if not forward_passed and (_float(row.get("regime_stability_score")) or 0.0) <= 0:
            reasons.append("regime_stability_not_positive_or_missing")
        if (_float(row.get("best_long_short_mean_bps")) or 0.0) <= 0:
            reasons.append("cost_adjusted_long_short_not_positive")
        if (_int(row.get("sample_count")) or 0) < 100:
            reasons.append("sample_count_insufficient")
        if not forward_passed:
            reasons.append("forward_validation_not_passed")
        review_after_forward_pass = forward_passed
        if review_after_forward_pass:
            reasons = sorted({*reasons, *STRATEGY_REVIEW_BLOCKING_REASONS})
        eligible: bool | str = not reasons
        recommended_action = (
            "REVIEW_FOR_ALPHA_FACTORY_STRATEGY"
            if review_after_forward_pass
            else (
                "CREATE_ALPHA_FACTORY_RESEARCH_CANDIDATE"
                if eligible
                else "DISPLAY_ONLY_FACTOR_REVIEW"
            )
        )
        if recommended_action == "REVIEW_FOR_ALPHA_FACTORY_STRATEGY":
            eligible = "strategy_review_pending"
        out.append(
            {
                "as_of_date": row.get("as_of_date"),
                "factor_id": factor_id,
                "factor_family": row.get("factor_family"),
                "correlation_cluster_id": row.get("correlation_cluster_id"),
                **_bridge_forward_context(forward_context_by_factor.get(factor_id, {})),
                "bridge_candidate_id": f"v5.factor_bridge.{factor_id}",
                "eligible_for_alpha_factory": eligible,
                "blocking_reasons": safe_json_dumps(reasons),
                "recommended_action": recommended_action,
                "live_order_effect": FACTOR_FACTORY_V2_LIVE_ORDER_EFFECT,
            }
        )
    for factor_id, row in sorted(forward_pass_rows_by_factor.items()):
        if factor_id in seen_factor_ids:
            continue
        out.append(
            {
                "as_of_date": row.get("as_of_date"),
                "factor_id": factor_id,
                "factor_family": row.get("factor_family"),
                "correlation_cluster_id": row.get("correlation_cluster_id"),
                **_bridge_forward_context(forward_context_by_factor.get(factor_id, {})),
                "bridge_candidate_id": f"v5.factor_bridge.{factor_id}",
                "eligible_for_alpha_factory": "strategy_review_pending",
                "blocking_reasons": safe_json_dumps(
                    sorted({"not_in_factor_paper_review_queue", *STRATEGY_REVIEW_BLOCKING_REASONS})
                ),
                "recommended_action": "REVIEW_FOR_ALPHA_FACTORY_STRATEGY",
                "live_order_effect": FACTOR_FACTORY_V2_LIVE_ORDER_EFFECT,
            }
        )
    out.extend(_fast_microstructure_strategy_bridge_rows(fast_microstructure_forward_test))
    out.sort(key=_bridge_candidate_rank_key)
    return _frame(out, FACTOR_STRATEGY_BRIDGE_CANDIDATE_FIELDS)


def _fast_microstructure_strategy_bridge_rows(
    frame: pl.DataFrame | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _rows(frame):
        if not _is_specific_fast_microstructure_forward_pass(row):
            continue
        feature_name = str(row.get("feature_name") or "").strip()
        symbol = str(row.get("symbol") or "").strip()
        regime = str(row.get("regime") or "").strip()
        horizon_hours = _int(row.get("horizon_hours"))
        if not feature_name or not symbol or not regime or horizon_hours is None:
            continue
        factor_id = f"fast_microstructure.{feature_name}"
        out.append(
            {
                "as_of_date": _fast_microstructure_as_of_date(row),
                "factor_id": factor_id,
                "factor_family": "fast_microstructure",
                "correlation_cluster_id": "fast_microstructure",
                "symbol": symbol,
                "regime": regime,
                "horizon": f"{horizon_hours}h",
                "horizon_hours": str(horizon_hours),
                "forward_sample_count": row.get("sample_count"),
                "forward_cost_adjusted_score": row.get("long_short_bps"),
                "bridge_candidate_id": _fast_microstructure_bridge_candidate_id(
                    feature_name=feature_name,
                    symbol=symbol,
                    regime=regime,
                    horizon_hours=horizon_hours,
                ),
                "eligible_for_alpha_factory": "strategy_review_pending",
                "blocking_reasons": safe_json_dumps(STRATEGY_REVIEW_BLOCKING_REASONS),
                "recommended_action": "REVIEW_FOR_ALPHA_FACTORY_STRATEGY",
                "live_order_effect": FACTOR_FACTORY_V2_LIVE_ORDER_EFFECT,
            }
        )
    return out


def _is_specific_fast_microstructure_forward_pass(row: dict[str, Any]) -> bool:
    if str(row.get("recommendation") or "") != "FORWARD_VALIDATION_PASS":
        return False
    regime = str(row.get("regime") or "").strip().upper()
    return bool(regime) and regime != FAST_MICROSTRUCTURE_FORWARD_AGGREGATE_REGIME


def _fast_microstructure_as_of_date(row: dict[str, Any]) -> Any:
    direct = row.get("as_of_date")
    if direct:
        return direct
    generated = str(row.get("generated_at") or "").strip()
    return generated[:10] if generated else None


def _fast_microstructure_bridge_candidate_id(
    *,
    feature_name: str,
    symbol: str,
    regime: str,
    horizon_hours: int,
) -> str:
    return ".".join(
        [
            "v5",
            "fast_microstructure_bridge",
            _slug(feature_name),
            _slug(symbol),
            _slug(regime),
            f"{horizon_hours}h",
        ]
    )


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    out = [character if character.isalnum() else "_" for character in text]
    return "_".join(part for part in "".join(out).split("_") if part) or "unknown"


def _forward_validation_pass_context_by_factor(
    frame: pl.DataFrame | None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(frame):
        factor_id = str(row.get("factor_id") or "")
        if not factor_id:
            continue
        if str(row.get("recommendation") or "") != "FORWARD_VALIDATION_PASS":
            continue
        grouped[factor_id].append(row)
    return {factor_id: _forward_validation_context(rows) for factor_id, rows in grouped.items()}


def _bridge_forward_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": "",
        "regime": "",
        "horizon": "",
        "horizon_hours": "",
        "forward_sample_count": None,
        "forward_cost_adjusted_score": None,
        **context,
    }


def _forward_validation_pass_by_factor(frame: pl.DataFrame | None) -> dict[str, bool]:
    return {
        factor_id: True
        for factor_id in _forward_validation_pass_rows_by_factor(frame)
    }


def _forward_validation_pass_rows_by_factor(
    frame: pl.DataFrame | None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _rows(frame):
        factor_id = str(row.get("factor_id") or "")
        if not factor_id:
            continue
        if str(row.get("recommendation") or "") != "FORWARD_VALIDATION_PASS":
            continue
        current = out.get(factor_id)
        if current is None or _forward_validation_rank_key(row) > _forward_validation_rank_key(
            current
        ):
            out[factor_id] = row
    return out


def _forward_validation_rank_key(row: dict[str, Any]) -> tuple[float, int, float]:
    return (
        _float(row.get("cost_adjusted_score")) or 0.0,
        _int(row.get("sample_count")) or 0,
        _float(row.get("rank_ic")) or 0.0,
    )


def _forward_validation_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(rows, key=_forward_validation_rank_key)
    horizon_hours = sorted(
        {
            hour
            for row in rows
            if (hour := _horizon_hour_value(row)) is not None
        }
    )
    return {
        "symbol": _join_unique(row.get("symbol") for row in rows),
        "regime": _join_unique(row.get("regime") or row.get("regime_state") for row in rows),
        "horizon": _format_horizon_label(horizon_hours)
        or _join_unique(row.get("horizon") for row in rows),
        "horizon_hours": ",".join(str(hour) for hour in horizon_hours),
        "forward_sample_count": best.get("sample_count"),
        "forward_cost_adjusted_score": best.get("cost_adjusted_score"),
    }


def _horizon_hour_value(row: dict[str, Any]) -> int | None:
    direct = _int(row.get("horizon_hours"))
    if direct is not None:
        return direct
    text = str(row.get("horizon") or "").strip().lower()
    digits = "".join(character for character in text if character.isdigit())
    return int(digits) if digits else None


def _format_horizon_label(hours: list[int]) -> str:
    if not hours:
        return ""
    if len(hours) == 1:
        return f"{hours[0]}h"
    if len(hours) == 2:
        return f"{hours[0]}h-{hours[1]}h"
    return ",".join(f"{hour}h" for hour in hours)


def _join_unique(values: Any) -> str:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return ",".join(out)


def _bridge_candidate_rank_key(row: dict[str, Any]) -> tuple[int, int, str]:
    action = str(row.get("recommended_action") or "")
    action_rank = {
        "REVIEW_FOR_ALPHA_FACTORY_STRATEGY": 0,
        "CREATE_ALPHA_FACTORY_RESEARCH_CANDIDATE": 1,
        "DISPLAY_ONLY_FACTOR_REVIEW": 2,
    }.get(action, 3)
    has_forward_context = 0 if str(row.get("symbol") or "").strip() else 1
    return (action_rank, has_forward_context, str(row.get("factor_id") or ""))


def _candidate_rank_key(row: dict[str, Any]) -> tuple[int, float, float, float, str]:
    return (
        1 if str(row.get("candidate_state") or "") == "PAPER_READY" else 0,
        _float(row.get("best_rank_ic_tstat")) or 0.0,
        _float(row.get("best_long_short_mean_bps")) or 0.0,
        _float(row.get("best_score")) or 0.0,
        str(row.get("factor_id") or ""),
    )


def _best_evidence_by_factor(evidence: pl.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _rows(evidence):
        factor_id = str(row.get("factor_id") or "")
        if not factor_id:
            continue
        current = out.get(factor_id)
        if current is None or _evidence_rank_key(row) > _evidence_rank_key(current):
            out[factor_id] = row
    return out


def _evidence_rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _float(row.get("score")) or 0.0,
        _float(row.get("rank_ic_tstat")) or 0.0,
        _float(row.get("long_short_mean_bps")) or 0.0,
    )


def _oos_score(candidate: dict[str, Any], evidence: dict[str, Any]) -> float | None:
    explicit = _float(
        candidate.get("oos_score")
        or candidate.get("recent_score")
        or evidence.get("oos_score")
        or evidence.get("recent_score")
    )
    if explicit is not None:
        return explicit
    rank_t = _float(candidate.get("best_rank_ic_tstat") or evidence.get("rank_ic_tstat"))
    spread = _float(
        candidate.get("best_long_short_mean_bps")
        or evidence.get("long_short_mean_bps")
    )
    if rank_t is None or spread is None:
        return None
    return round(min(rank_t, 5.0) + spread / 100.0, 6)


def _regime_stability_score(factor_id: str, evidence: pl.DataFrame) -> float | None:
    regime_column = _first_existing_column(evidence, ("regime", "regime_state", "market_regime"))
    if not regime_column:
        return None
    rows = [
        row
        for row in _rows(evidence)
        if str(row.get("factor_id") or "") == factor_id
        and str(row.get(regime_column) or "").strip()
    ]
    if not rows:
        return None
    positive = sum(1 for row in rows if (_float(row.get("long_short_mean_bps")) or 0.0) > 0)
    return round(positive / len(rows), 6)


def _regime_recommendation(row: dict[str, Any]) -> str:
    samples = _int(row.get("valid_sample_count") or row.get("sample_count")) or 0
    spread = _float(row.get("long_short_mean_bps")) or 0.0
    rank_ic = _float(row.get("rank_ic_mean")) or 0.0
    if samples < 30:
        return "RESEARCH_MORE_SAMPLES"
    if spread > 0 and rank_ic > 0:
        return "REGIME_EFFECTIVE_KEEP_SHADOW"
    if spread < 0 and rank_ic < 0:
        return "REGIME_WEAK_OR_NEGATIVE"
    return "REGIME_MIXED_RESEARCH"


def _external_research_term(term: str) -> bool:
    return term in {
        "market_pressure_state",
        "bottom_zone_score",
        "taker_sell_exhaustion",
        "late_breakout_failure_score",
        "distance_to_vwap",
        "spread_normalization",
    }


def _latest_as_of_date(rows: list[dict[str, Any]]) -> str:
    values = sorted(str(row.get("as_of_date") or "") for row in rows if row.get("as_of_date"))
    return values[-1] if values else ""


def _first_existing_column(frame: pl.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _rows(frame: pl.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.is_empty():
        return []
    return frame.to_dicts()


def _frame(rows: list[dict[str, Any]], fields: list[str]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={field: pl.Utf8 for field in fields})
    return pl.DataFrame(rows, infer_schema_length=None).select(fields)


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
