import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from quant_lab.contracts.models import CostEstimate, RiskPermission
from quant_lab.contracts.v5_quant_lab import (
    RISK_PERMISSION_CONTRACT_VERSION,
    V5_COST_ESTIMATE_RESPONSE_SCHEMA_VERSION,
    V5_QUANT_LAB_CONTRACT_VERSION,
    V5_RISK_PERMISSION_RESPONSE_SCHEMA_VERSION,
    V5_TELEMETRY_DATASET_SCHEMA_VERSION,
)
from quant_lab.strategy_telemetry import ingest
from quant_lab.symbols import normalize_symbol

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "v5_quant_lab_contract.yaml"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "v5_contract"


def test_contract_runtime_versions_match_yaml():
    contract = _load_contract()

    assert contract["contract_version"] == V5_QUANT_LAB_CONTRACT_VERSION
    assert (
        contract["schemas"]["cost_estimate_response"]["schema_version"]
        == V5_COST_ESTIMATE_RESPONSE_SCHEMA_VERSION
    )
    assert (
        contract["schemas"]["risk_permission_response"]["schema_version"]
        == V5_RISK_PERMISSION_RESPONSE_SCHEMA_VERSION
    )
    assert (
        contract["datasets"]["v5_telemetry"]["schema_version"]
        == V5_TELEMETRY_DATASET_SCHEMA_VERSION
        == ingest.SCHEMA_VERSION
    )
    assert CostEstimate(
        symbol="BNB/USDT",
        regime="Trending",
        notional_usdt=1000.0,
        fallback_level="GLOBAL_DEFAULT",
        total_cost_bps=25.0,
        cost_bps=25.0,
    ).schema_version == V5_COST_ESTIMATE_RESPONSE_SCHEMA_VERSION
    assert RiskPermission(
        strategy="v5",
        version="5.0.0",
        permission="ABORT",
        max_gross_exposure=0.0,
        max_single_weight=0.0,
        cost_model_version="test",
        gate_version="test",
        created_at=datetime(2026, 5, 14, tzinfo=UTC),
    ).contract_version == RISK_PERMISSION_CONTRACT_VERSION


def test_cost_contract_fixtures_validate_required_fields_enums_and_timestamps():
    contract = _load_contract()
    cases = [
        ("cost_estimate_request", _load_json("cost_request_bnb_usdt_trending.json")),
        ("cost_estimate_response", _load_json("cost_response_bnb_usdt_public_proxy.json")),
        ("cost_estimate_response", _load_json("cost_response_global_default_degraded.json")),
    ]

    for schema_name, payload in cases:
        _validate_payload(contract, schema_name, payload)


def test_risk_permission_contract_fixtures_validate_required_fields_enums_and_timestamps():
    contract = _load_contract()
    for fixture in [
        "risk_permission_active_abort.json",
        "risk_permission_stale_abort.json",
    ]:
        _validate_payload(contract, "risk_permission_response", _load_json(fixture))


def test_telemetry_contract_fixtures_validate_required_fields_enums_and_timestamps():
    contract = _load_contract()

    success_request = _load_jsonl("quant_lab_request_success_200.jsonl")[0]
    timeout_fallback = _load_jsonl("quant_lab_timeout_fallback.jsonl")[0]
    _validate_payload(contract, "quant_lab_request_event", success_request)
    _validate_payload(contract, "quant_lab_fallback_event", timeout_fallback)

    assert success_request["status_code"] == 200
    assert success_request["success"] is True
    assert success_request["fallback_used"] is False
    assert timeout_fallback["fallback_used"] is True
    assert timeout_fallback["error_type"] == "QuantLabTimeout"


def test_trade_fill_summary_fixture_validates_contract():
    contract = _load_contract()
    schema = contract["schemas"]["trade_fill_summary"]
    rows = _load_csv("trades_bnb_buy_sell.csv")

    assert {row["side"] for row in rows} == {"buy", "sell"}
    for row in rows:
        assert row["schema_version"] == schema["schema_version"]
        _validate_payload(contract, "trade_fill_summary", row)
        assert row["normalized_symbol"] == normalize_symbol(row["symbol"])


def test_symbol_normalization_contract_examples():
    assert normalize_symbol("BNB/USDT") == "BNB-USDT"
    assert normalize_symbol("BNB-USDT") == "BNB-USDT"
    assert normalize_symbol("BNBUSDT") == "BNB-USDT"
    assert normalize_symbol("OKX:BNB-USDT") == "BNB-USDT"


def test_global_default_cost_fixture_is_explicitly_degraded():
    payload = _load_json("cost_response_global_default_degraded.json")

    assert payload["cost_source"] == "global_default"
    assert payload["source"] == "global_default"
    assert payload["fallback_level"] == "GLOBAL_DEFAULT"
    assert payload["degraded_reason"] == "global_default_cost"
    assert payload["fallback_reason"] in {"symbol_missing", "service_unavailable"}


def _load_contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def _load_json(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _load_jsonl(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (FIXTURE_DIR / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_csv(name: str) -> list[dict[str, str]]:
    with (FIXTURE_DIR / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_payload(contract: dict, schema_name: str, payload: dict) -> None:
    schema = contract["schemas"][schema_name]
    assert payload["schema_version"] == schema["schema_version"]

    nullable = set(schema.get("nullable", []))
    missing = [
        field
        for field in schema["required"]
        if field not in payload or (payload[field] is None and field not in nullable)
    ]
    assert missing == []

    for field, allowed_values in schema.get("enums", {}).items():
        if field in payload and payload[field] is not None:
            assert str(payload[field]) in {str(value) for value in allowed_values}

    for field in schema.get("timestamps", []):
        if payload.get(field) is not None:
            _assert_utc_iso8601(str(payload[field]))


def _assert_utc_iso8601(value: str) -> None:
    assert value.endswith("Z") or value.endswith("+00:00")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.utcoffset() == UTC.utcoffset(parsed)

