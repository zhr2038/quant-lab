from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Any

from quant_lab.contracts.models import require_utc
from quant_lab.decision.contracts import (
    EXPERIMENT_VERSION,
    Advice,
    CostObservation,
    Distribution,
    HourBar,
    InputSnapshot,
)
from quant_lab.decision.current_cost_contracts import CurrentCostObservation

HOUR = timedelta(hours=1)
MIN_SAMPLES = 40
MIN_TAIL_SAMPLES = 12


def content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def advice_identity(value: Advice) -> str:
    payload = value.model_dump(mode="json")
    payload.pop("advice_id")
    return "advice-" + content_hash(payload)


def prepare_history(bars: list[HourBar], now: datetime) -> list[HourBar]:
    """Select only available closed bars; reject conflicting duplicate identities."""
    now = require_utc(now)
    selected: dict[tuple[str, datetime], HourBar] = {}
    for bar in bars:
        if bar.ts + HOUR > now or bar.ingest_ts > now:
            continue
        key = (bar.symbol, bar.ts)
        previous = selected.get(key)
        if previous is not None:
            fields = ("open", "high", "low", "close", "volume")
            if any(getattr(previous, field) != getattr(bar, field) for field in fields):
                raise ValueError(f"conflicting duplicate bar: {bar.symbol} {bar.ts}")
            if previous.ingest_ts >= bar.ingest_ts:
                continue
        selected[key] = bar
    return sorted(selected.values(), key=lambda row: (row.symbol, row.ts))


def contiguous(bars: list[HourBar]) -> bool:
    return bool(bars) and all(
        right.ts - left.ts == HOUR for left, right in zip(bars, bars[1:], strict=False)
    )


def context_at(bars: list[HourBar], index: int) -> tuple[str, float, float] | None:
    if index < 24:
        return None
    window = bars[index - 24 : index + 1]
    if not contiguous(window):
        return None
    trend = (window[-1].close / window[0].close - 1) * 10_000
    returns = [math.log(b.close / a.close) for a, b in zip(window, window[1:], strict=False)]
    volatility = statistics.pstdev(returns) * math.sqrt(24) * 10_000
    return ("trend_up" if trend >= 0 else "trend_down", trend, volatility)


def historical_labels(
    bars: list[HourBar], *, horizon: int, context: str
) -> tuple[list[tuple[datetime, datetime, float]], int]:
    """One full hour after signal availability; UTC grid prevents overlapping labels.

    This reconstructs historical asset paths. It is deliberately not a V5 portfolio
    backtest and does not pretend historical signals were actually published.
    """
    labels: list[tuple[datetime, datetime, float]] = []
    matches = 0
    for index in range(24, len(bars) - horizon - 2):
        state = context_at(bars, index)
        if state is None or state[0] != context:
            continue
        entry_index, exit_index = index + 2, index + 2 + horizon
        if not contiguous(bars[index : exit_index + 1]):
            continue
        matches += 1
        hour_index = int(bars[index].ts.timestamp()) // 3_600
        if hour_index % (horizon + 1):
            continue
        gross = (bars[exit_index].open / bars[entry_index].open - 1) * 10_000
        labels.append((bars[index].ts + HOUR, bars[exit_index].ts, gross))
    return labels, matches


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    left = int(index)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (index - left)


def describe(
    bars: list[HourBar], *, horizon: int, context: str, cost_bps: float | None
) -> Distribution:
    labels, matches = historical_labels(bars, horizon=horizon, context=context)
    # The last 30% is a chronological diagnostic, not an untouched holdout claim.
    # Purge outcomes crossing the split and embargo the next horizon.
    split = bars[0].ts + (bars[-1].ts - bars[0].ts) * 0.7 if bars else None
    train = [row for row in labels if split is not None and row[1] <= split]
    tail = [
        row
        for row in labels
        if split is not None and row[0] >= split + timedelta(hours=horizon + 1)
    ]
    selected = sorted(train + tail)
    gross = [row[2] for row in selected]
    data: dict[str, Any] = {
        "samples": matches,
        "non_overlapping_samples": len(selected),
        "calendar_days": len({row[0].date() for row in selected}),
        "first_signal_at": selected[0][0] if selected else None,
        "last_signal_at": selected[-1][0] if selected else None,
        "gross_mean_bps": statistics.fmean(gross) if gross else None,
        "gross_p10_bps": quantile(gross, 0.1),
        "gross_p50_bps": quantile(gross, 0.5),
        "gross_p90_bps": quantile(gross, 0.9),
        "chronological_tail_samples": len(tail),
    }
    if gross and cost_bps is not None:
        for name in ("mean", "p10", "p50", "p90"):
            data[f"net_{name}_bps"] = data[f"gross_{name}_bps"] - cost_bps
        data["double_cost_mean_bps"] = data["gross_mean_bps"] - 2 * cost_bps
        data["historical_positive_fraction"] = sum(x > cost_bps for x in gross) / len(gross)
        if tail:
            data["chronological_tail_net_mean_bps"] = (
                statistics.fmean(row[2] for row in tail) - cost_bps
            )
    return Distribution(**data)


def build_advice(
    history: list[HourBar], *, inputs: InputSnapshot, symbol: str, horizon: int, now: datetime
) -> Advice:
    now = require_utc(now)
    bars = prepare_history([bar for bar in history if bar.symbol == symbol], now)
    latest = max(
        (bar for bar in inputs.bars if bar.symbol == symbol), key=lambda b: b.ts, default=None
    )
    cost = next((item for item in inputs.costs if item.symbol == symbol), None)
    if cost is None:
        cost = CostObservation(
            symbol=symbol,
            source="missing",
            quality="missing",
            version="none",
            missing_reasons=["COST_MISSING"],
        )
    market_asof = latest.ts + HOUR if latest else None
    entry_at = market_asof + HOUR if market_asof else None
    context = context_at(bars, len(bars) - 1) if bars else None
    missing: list[str] = []
    if latest is None:
        missing.append("CURRENT_MARKET_MISSING")
    elif not bars or bars[-1].ts != latest.ts:
        missing.append("HISTORY_CURRENT_MISMATCH")
    if entry_at is not None and now >= entry_at:
        missing.append("CURRENT_MARKET_STALE")
    if context is None:
        missing.append("CONTEXT_WINDOW_INCOMPLETE")
    if cost.roundtrip_bps is None:
        missing.append("COST_MISSING")
    current_cost_expiry = (
        cost.current.valid_until if isinstance(cost, CurrentCostObservation) else None
    )
    if isinstance(cost, CurrentCostObservation):
        if current_cost_expiry is None or now >= current_cost_expiry:
            missing.append("CURRENT_COST_EXPIRED_OR_UNAVAILABLE")
    elif cost.as_of is None or now - cost.as_of > timedelta(days=2):
        missing.append("COST_STALE_OR_UNDATED")
    if cost.as_of is not None and cost.as_of > now:
        missing.append("COST_TIMESTAMP_FUTURE")
    state = context[0] if context else "unavailable"
    distribution = describe(bars, horizon=horizon, context=state, cost_bps=cost.roundtrip_bps)
    if distribution.non_overlapping_samples < MIN_SAMPLES:
        missing.append("HISTORICAL_SAMPLE_INSUFFICIENT")
    if distribution.chronological_tail_samples < MIN_TAIL_SAMPLES:
        missing.append("RECENT_DIAGNOSTIC_SAMPLE_INSUFFICIENT")
    reasons = list(missing)
    action = "NO_VIEW"
    explanation = "当前不形成方向观点；请查看缺失原因，按 V5 原有规则处理。"
    if "CURRENT_COST_EXPIRED_OR_UNAVAILABLE" in missing:
        explanation = (
            "当前账户费率或盘口成本不可用或已过期，暂不形成方向观点。"
            "等待下一次只读采样；历史探针仅保留作核对依据。"
        )
    if "COST_STALE_OR_UNDATED" in missing:
        explanation = (
            "先更新只读成交成本并核对样本，再复核入场候选。当前成本依据过旧或缺少时间，"
            "下方数值仅用于历史成本场景比较，暂不形成可用方向观点。"
        )
    if not missing:
        mean = distribution.net_mean_bps
        tail_mean = distribution.chronological_tail_net_mean_bps
        assert mean is not None and tail_mean is not None
        if mean <= 0 and tail_mean <= 0:
            action = "DEFER"
            reasons.append("HISTORICAL_COST_DRAG")
            explanation = (
                "相似行情与近期诊断窗口在当前成本假设下的平均结果均不为正，"
                "研究参考为等待新窗口。历史统计尚未证明实际交易增益。"
            )
        elif mean > 0 and tail_mean > 0 and (distribution.double_cost_mean_bps or 0) > 0:
            if cost.trusted_for_paper:
                action = "REVIEW_ENTRY"
                reasons.append("POSITIVE_HISTORICAL_REFERENCE")
                explanation = (
                    "相似行情在当前及双倍成本假设下存在正向历史参考，"
                    "可交由 V5 原规则复核候选；这不是已验证的入场信号。"
                )
            else:
                action = "KEEP_BASELINE"
                reasons.append("COST_REQUIRES_CALIBRATION")
                explanation = "历史参考偏正，但成本证据尚不足；保持 V5 原规则，先核对成交成本。"
        else:
            action = "KEEP_BASELINE"
            reasons.append("HISTORICAL_REFERENCE_MIXED")
            explanation = "历史窗口、近期窗口或成本压力结果不一致，保持 V5 原规则。"
    if not cost.trusted_for_paper:
        reasons.append("COST_NOT_PAPER_TRUSTED")
    if isinstance(cost, CurrentCostObservation):
        reasons.append("CURRENT_COST_IS_ESTIMATE")
    reasons.append("FORWARD_VALUE_NOT_ESTABLISHED")
    data_hash = content_hash([bar.model_dump(mode="json") for bar in bars])
    opportunity = content_hash(
        {"symbol": symbol, "market_asof": market_asof, "experiment": EXPERIMENT_VERSION}
    )
    identity = content_hash(
        {
            "opportunity": opportunity,
            "horizon": horizon,
            "input": inputs.snapshot_id,
            "data": data_hash,
        }
    )
    expiry = entry_at if entry_at and now < entry_at else now
    if current_cost_expiry is not None:
        expiry = max(now, min(expiry, current_cost_expiry))
    advice = Advice(
        advice_id="advice-" + identity,
        opportunity_id="opportunity-" + opportunity,
        symbol=symbol,
        horizon_hours=horizon,
        generated_at=now,
        market_asof=market_asof,
        expires_at=expiry,
        reference_entry_at=entry_at,
        reference_exit_at=entry_at + timedelta(hours=horizon) if entry_at else None,
        action=action,
        reason_codes=reasons,
        explanation=explanation,
        invalidation_conditions=["参考到期", "新的闭合小时 K 线出现", "成本或数据质量发生变化"],
        context=state,
        last_close=latest.close if latest else None,
        trend_24h_bps=context[1] if context else None,
        volatility_24h_bps=context[2] if context else None,
        cost=cost,
        distribution=distribution,
        input_snapshot_id=inputs.snapshot_id,
        data_snapshot_hash=data_hash,
    )
    return advice.model_copy(update={"advice_id": advice_identity(advice)})
