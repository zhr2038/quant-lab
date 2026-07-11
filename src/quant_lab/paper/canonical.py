from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from quant_lab.paper.contracts import PaperRule

_FIELD_ALIASES = {
    "px_close": "close",
    "close_px": "close",
    "last": "close",
    "qty": "volume",
    "base_volume": "volume",
    "ret": "return",
    "returns": "return",
}


def normalize_timeframe(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "")
    match = re.fullmatch(r"([1-9][0-9]*)(min|m|hour|hours|hr|h|day|days|d)", text)
    if not match:
        return text
    count = int(match.group(1))
    unit = match.group(2)
    if unit in {"min", "m"}:
        if count % 1440 == 0:
            return f"{count // 1440}d"
        if count % 60 == 0:
            return f"{count // 60}h"
        return f"{count}m"
    if unit in {"hour", "hours", "hr", "h"}:
        return f"{count}h"
    return f"{count}d"


def canonical_event_id(
    *,
    symbol: Any,
    timeframe: Any,
    bar_open_ts: Any,
    signal_family: Any,
    entry_rule: PaperRule | Mapping[str, Any],
    source_dataset_version: Any,
) -> str:
    rule = entry_rule if isinstance(entry_rule, PaperRule) else PaperRule.model_validate(entry_rule)
    payload = {
        "symbol": _normalize_symbol(symbol),
        "timeframe": normalize_timeframe(timeframe),
        "bar_open_ts": str(bar_open_ts or "").strip(),
        "signal_family": str(signal_family or "").strip().lower(),
        "entry_rule": rule.model_dump(mode="json", exclude_none=True),
        "source_dataset_version": str(source_dataset_version or "").strip(),
    }
    return _stable_id("evt", payload)


def canonical_market_event_id(event: Mapping[str, Any]) -> str:
    """Build a strategy-independent identity for legacy candidate events."""

    symbol = _normalize_symbol(event.get("symbol") or event.get("v5_symbol"))
    timeframe = normalize_timeframe(event.get("timeframe") or event.get("timeframe_main") or "1h")
    bar_open_ts = (
        event.get("bar_open_ts")
        or event.get("window_end_ts")
        or event.get("decision_ts")
        or event.get("ts_utc")
    )
    source_event = event.get("source_event_id") or event.get("market_event_id") or ""
    family = _normalized_signal_family(
        event.get("signal_family") or event.get("strategy_candidate") or event.get("strategy_id")
    )
    entry_condition = (
        event.get("normalized_entry_condition")
        or event.get("entry_rule")
        or {
            key: event.get(key)
            for key in (
                "rank",
                "alpha6_decision",
                "regime",
                "risk_level",
                "cost_gate_verified",
            )
            if event.get(key) not in (None, "")
        }
    )
    return _stable_id(
        "evt",
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "bar_open_ts": _normalized_bar_open_ts(bar_open_ts, timeframe),
            "source_event": str(source_event),
            "signal_family": family,
            "entry_condition": _normalize_json(entry_condition),
            "source_dataset_version": str(
                event.get("source_dataset_version")
                or event.get("schema_version")
                or event.get("source_version")
                or "legacy"
            ),
        },
    )


def strategy_evaluation_id(
    *,
    event_id: str,
    strategy_id: Any,
    strategy_version: Any,
    source_event_id: Any = None,
) -> str:
    return _stable_id(
        "eval",
        {
            "canonical_event_id": event_id,
            "strategy_id": str(strategy_id or "").strip(),
            "strategy_version": str(strategy_version or "").strip(),
            "source_event_id": str(source_event_id or "").strip(),
        },
    )


def annotate_shared_events(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(rows):
        row = dict(source)
        event_id = str(row.get("canonical_event_id") or "")
        evaluation_id = str(row.get("strategy_evaluation_id") or "") or _stable_id(
            "eval",
            {
                "canonical_event_id": event_id,
                "strategy_id": row.get("strategy_id"),
                "strategy_version": row.get("strategy_version"),
                "fallback_index": index if not event_id else None,
            },
        )
        current = deduplicated.get(evaluation_id)
        if current is None or (_has_observed_outcome(row) and not _has_observed_outcome(current)):
            row["strategy_evaluation_id"] = evaluation_id
            deduplicated[evaluation_id] = row
    materialized = list(deduplicated.values())
    counts = Counter(str(row.get("canonical_event_id") or "") for row in materialized)
    for row in materialized:
        event_id = str(row.get("canonical_event_id") or "")
        shared_count = max(counts.get(event_id, 1), 1)
        row["shared_event_group_id"] = event_id
        row["shared_event_strategy_count"] = shared_count
        row["event_independence_weight"] = 1.0 / shared_count
    return materialized


def _has_observed_outcome(row: Mapping[str, Any]) -> bool:
    return any(
        row.get(field) not in (None, "")
        for field in ("after_cost_bps", "net_pnl_bps", "paper_pnl_bps", "net_bps")
    )


def _normalized_bar_open_ts(value: Any, timeframe: str) -> str:
    parsed: datetime | None = None
    try:
        if isinstance(value, (int, float)) or str(value).strip().isdigit():
            raw = float(value)
            if raw > 10_000_000_000:
                raw /= 1_000.0
            parsed = datetime.fromtimestamp(raw, tz=UTC)
        elif value not in (None, ""):
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            parsed = parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (ValueError, TypeError, OSError):
        parsed = None
    if parsed is None:
        return str(value or "").strip()
    seconds = _timeframe_seconds(timeframe)
    if seconds is None:
        return parsed.isoformat().replace("+00:00", "Z")
    floored = int(parsed.timestamp()) // seconds * seconds
    return datetime.fromtimestamp(floored, tz=UTC).isoformat().replace("+00:00", "Z")


def _timeframe_seconds(timeframe: str) -> int | None:
    match = re.fullmatch(r"([1-9][0-9]*)(m|h|d)", timeframe)
    if not match:
        return None
    multiplier = {"m": 60, "h": 3_600, "d": 86_400}[match.group(2)]
    return int(match.group(1)) * multiplier


def factor_semantic_identity(
    *,
    formula: str | Mapping[str, Any],
    feature_dependencies: Iterable[Any] = (),
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_formula = _normalize_formula(formula)
    dependencies = sorted(
        {
            _FIELD_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
            for value in feature_dependencies
            if str(value).strip()
        }
    )
    operator_graph_hash = _stable_hash(normalized_formula)
    payload = {
        "formula": normalized_formula,
        "dependencies": dependencies,
        "parameters": _normalize_json(parameters or {}),
    }
    formula_hash = _stable_hash(payload)
    return {
        "canonical_factor_id": f"factor:{formula_hash[:20]}",
        "factor_formula_hash": formula_hash,
        "feature_dependencies": dependencies,
        "operator_graph_hash": operator_graph_hash,
    }


def annotate_factor_lineage(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    canonical_owner: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for raw in rows:
        row = dict(raw)
        identity = factor_semantic_identity(
            formula=row.get("formula") or row.get("expression") or row.get("operator_graph") or {},
            feature_dependencies=_as_list(
                row.get("feature_dependencies") or row.get("dependencies")
            ),
            parameters=row.get("parameters") if isinstance(row.get("parameters"), Mapping) else {},
        )
        row.update(identity)
        canonical = identity["canonical_factor_id"]
        original = str(row.get("factor_id") or row.get("name") or canonical)
        owner = canonical_owner.setdefault(canonical, original)
        row["duplicate_of"] = "" if owner == original else owner
        row.setdefault("correlation_cluster_id", canonical)
        counts[canonical] += 1
        materialized.append(row)
    for row in materialized:
        count = max(counts[row["canonical_factor_id"]], 1)
        row["effective_independence_weight"] = 1.0 / count
    return materialized


def _normalize_formula(formula: str | Mapping[str, Any]) -> Any:
    if isinstance(formula, Mapping):
        return _normalize_json(formula)
    text = str(formula or "").strip()
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError):
        return {"literal": " ".join(text.lower().split())}
    return _normalize_ast(tree.body)


def _normalize_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        return {"field": _FIELD_ALIASES.get(node.id.lower(), node.id.lower())}
    if isinstance(node, ast.Constant):
        return {"constant": node.value}
    if isinstance(node, ast.Attribute):
        return {"attribute": ast.unparse(node).lower()}
    if isinstance(node, ast.Call):
        return {
            "call": _normalize_ast(node.func),
            "args": [_normalize_ast(arg) for arg in node.args],
            "keywords": sorted(
                ((kw.arg or "", _normalize_ast(kw.value)) for kw in node.keywords),
                key=lambda item: item[0],
            ),
        }
    if isinstance(node, ast.BinOp):
        operator = type(node.op).__name__
        values = [_normalize_ast(node.left), _normalize_ast(node.right)]
        if isinstance(node.op, (ast.Add, ast.Mult)):
            values.sort(key=_json_key)
        return {"binary": operator, "values": values}
    if isinstance(node, ast.UnaryOp):
        return {"unary": type(node.op).__name__, "value": _normalize_ast(node.operand)}
    if isinstance(node, ast.Compare):
        return {
            "compare": [
                _normalize_ast(node.left),
                *[_normalize_ast(item) for item in node.comparators],
            ],
            "operators": [type(item).__name__ for item in node.ops],
        }
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_normalize_ast(item) for item in node.elts]
        if isinstance(node, ast.Set):
            values.sort(key=_json_key)
        return {type(node).__name__.lower(): values}
    return {"ast": ast.dump(node, annotate_fields=True, include_attributes=False)}


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_json(item) for item in value]
    return value


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [item.strip() for item in value.split(",") if item.strip()]
        return parsed if isinstance(parsed, list) else [parsed]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper().replace("_", "-").replace("/", "-")


def _normalized_signal_family(value: Any) -> str:
    text = str(value or "unknown").strip().lower().replace("-", "_")
    if re.search(r"(^|[._])f[34]([._]|$)", text):
        return "f3_f4_shared_entry"
    return text


def _stable_id(prefix: str, payload: Any) -> str:
    return f"{prefix}:{_stable_hash(payload)[:32]}"


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_json_key(payload).encode("utf-8")).hexdigest()


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
