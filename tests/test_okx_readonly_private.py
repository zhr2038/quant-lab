from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from quant_lab.costs.model import build_cost_bucket_daily_inputs
from quant_lab.data.lake import read_parquet_dataset
from quant_lab.ingest.okx_readonly_private import (
    ACCOUNT_BILLS_ENDPOINT,
    FILLS_HISTORY_ENDPOINT,
    OKXReadOnlyAPIError,
    OKXReadOnlyClient,
    OKXReadOnlyConfig,
    OKXReadOnlySafetyError,
    OKXReadOnlyTimeout,
    normalize_okx_bills,
    normalize_okx_fills,
    publish_okx_bills_to_lake,
    publish_okx_fills_to_lake,
)

API_KEY = "read-key-123"
SECRET_KEY = "secret-key-456"
PASSPHRASE = "passphrase-789"


def test_private_readonly_request_uses_get_and_auth_headers(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_request(self, method, url, params=None, headers=None):
        captured["method"] = method
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        request = httpx.Request(method, str(self.base_url.join(url)), params=params)
        return httpx.Response(
            200,
            json={"code": "0", "msg": "", "data": [_raw_fill()]},
            request=request,
        )

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    client = OKXReadOnlyClient(_config())

    fills = client.get_fills_history("SPOT", inst_id="BTC-USDT", limit=1)

    assert fills == [_raw_fill()]
    assert captured["method"] == "GET"
    assert captured["url"] == FILLS_HISTORY_ENDPOINT
    assert captured["params"] == {"instType": "SPOT", "instId": "BTC-USDT", "limit": "1"}
    headers = captured["headers"]
    assert headers[_ok_header("KEY")] == API_KEY
    assert headers[_ok_header("PASSPHRASE")] == PASSPHRASE
    assert headers[_ok_header("SIGN")]
    assert headers[_ok_header("TIMESTAMP")]


def test_endpoint_allowlist_and_forbidden_method_are_enforced():
    client = OKXReadOnlyClient(_config())

    with pytest.raises(OKXReadOnlySafetyError, match="only allows GET"):
        client._request_private("POST", FILLS_HISTORY_ENDPOINT)

    with pytest.raises(OKXReadOnlySafetyError, match="allowlist"):
        client._request_private("GET", "/api/v5/asset/withdrawal")

    with pytest.raises(OKXReadOnlySafetyError, match="allowlist"):
        client._request_private("GET", "/api/v5/trade/order")


def test_account_bills_and_config_use_allowed_get_endpoints(monkeypatch):
    urls: list[str] = []

    def fake_request(self, method, url, params=None, headers=None):
        urls.append(url)
        request = httpx.Request(method, str(self.base_url.join(url)), params=params)
        data = [_raw_bill()] if url == ACCOUNT_BILLS_ENDPOINT else [{"acctLv": "1"}]
        return httpx.Response(200, json={"code": "0", "msg": "", "data": data}, request=request)

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    client = OKXReadOnlyClient(_config())

    assert client.get_account_bills(ccy="USDT") == [_raw_bill()]
    assert client.get_account_config() == {"acctLv": "1"}
    assert urls == [ACCOUNT_BILLS_ENDPOINT, "/api/v5/account/config"]


def test_nonzero_okx_code_raises_redacted_error(monkeypatch):
    def fake_request(self, method, url, params=None, headers=None):
        request = httpx.Request(method, str(self.base_url.join(url)), params=params)
        return httpx.Response(
            200,
            json={"code": "50113", "msg": f"bad {SECRET_KEY}", "data": []},
            request=request,
        )

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    client = OKXReadOnlyClient(_config())

    with pytest.raises(OKXReadOnlyAPIError) as exc_info:
        client.get_fills_history("SPOT")

    error_text = str(exc_info.value)
    assert "50113" in error_text
    assert SECRET_KEY not in error_text
    assert "<redacted>" in error_text


def test_timeout_raises_clear_redacted_error(monkeypatch):
    def fake_request(self, method, url, params=None, headers=None):
        raise httpx.TimeoutException(f"timeout {PASSPHRASE}")

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    client = OKXReadOnlyClient(_config(max_retries=0))

    with pytest.raises(OKXReadOnlyTimeout) as exc_info:
        client.get_fills_history("SPOT")

    assert PASSPHRASE not in str(exc_info.value)


def test_fills_standardization():
    fills = normalize_okx_fills([_raw_fill()])

    assert len(fills) == 1
    fill = fills[0]
    assert fill.venue == "okx"
    assert fill.inst_type == "SPOT"
    assert fill.inst_id == "BTC-USDT"
    assert fill.trade_id == "trade-1"
    assert fill.order_id == "order-1"
    assert fill.side == "buy"
    assert fill.fill_price == 100
    assert fill.fill_size == 2
    assert fill.fee == -0.1
    assert fill.fee_currency == "USDT"
    assert fill.liquidity == "T"
    assert fill.ts == datetime(2026, 2, 16, tzinfo=UTC)
    assert fill.source == "okx_readonly_private"


def test_bills_standardization():
    bills = normalize_okx_bills([_raw_bill()])

    assert len(bills) == 1
    bill = bills[0]
    assert bill.venue == "okx"
    assert bill.bill_id == "bill-1"
    assert bill.ccy == "USDT"
    assert bill.amount == -0.1
    assert bill.balance == 999.9
    assert bill.bill_type == "2"
    assert bill.sub_type == "1"
    assert bill.ts == datetime(2026, 2, 16, tzinfo=UTC)
    assert bill.source == "okx_readonly_private"


def test_publish_private_readonly_data_does_not_write_secrets_to_parquet(tmp_path, caplog):
    raw_fill = {
        **_raw_fill(),
        "apiKeyEcho": API_KEY,
        "secretEcho": SECRET_KEY,
        "passphraseEcho": PASSPHRASE,
        "signEcho": "signature-value",
    }
    raw_bill = {
        **_raw_bill(),
        "apiKeyEcho": API_KEY,
        "secretEcho": SECRET_KEY,
        "passphraseEcho": PASSPHRASE,
    }

    fill_result = publish_okx_fills_to_lake([raw_fill], tmp_path / "lake")
    bill_result = publish_okx_bills_to_lake([raw_bill], tmp_path / "lake")

    assert fill_result == {"bronze_fills_rows": 1, "fill_event_rows": 1}
    assert bill_result == {"bronze_bills_rows": 1, "account_bill_rows": 1}
    fills_history_path = tmp_path / "lake" / "bronze" / "okx_private_readonly" / "fills_history"
    bills_path = tmp_path / "lake" / "bronze" / "okx_private_readonly" / "bills"
    assert list(fills_history_path.rglob("*.parquet"))
    assert list((tmp_path / "lake" / "silver" / "fill_event").rglob("*.parquet"))
    assert list((tmp_path / "lake" / "silver" / "account_bill").rglob("*.parquet"))

    parquet_text = "\n".join(
        frame.write_json()
        for frame in [
            read_parquet_dataset(fills_history_path),
            read_parquet_dataset(bills_path),
            read_parquet_dataset(tmp_path / "lake" / "silver" / "fill_event"),
            read_parquet_dataset(tmp_path / "lake" / "silver" / "account_bill"),
        ]
    )
    log_text = caplog.text
    for secret in [API_KEY, SECRET_KEY, PASSPHRASE, "signature-value"]:
        assert secret not in parquet_text
        assert secret not in log_text


def test_fills_can_generate_cost_bucket_daily_input():
    cost_rows = build_cost_bucket_daily_inputs(normalize_okx_fills([_raw_fill()]))

    assert cost_rows == [
        {
            "symbol": "BTC-USDT",
            "cost_day": "2026-02-16",
            "regime": "realized",
            "notional_usdt": 200.0,
            "fee_abs": 0.1,
            "source": "okx_readonly_private",
            "cost_bps": 5.0,
        }
    ]


def test_forbidden_endpoint_literals_do_not_appear_in_implementation():
    implementation_text = Path(
        "src/quant_lab/ingest/okx_readonly_private.py"
    ).read_text(encoding="utf-8")
    forbidden = [
        "/api/v5/trade/cancel-order",
        "/api/v5/trade/amend-order",
        "/api/v5/asset/withdrawal",
        "place_order",
        "cancel_order",
        "amend_order",
        "transfer_funds",
    ]

    for keyword in forbidden:
        assert keyword not in implementation_text


def _config(max_retries: int = 3) -> OKXReadOnlyConfig:
    return OKXReadOnlyConfig(
        api_key=API_KEY,
        secret_key=SECRET_KEY,
        passphrase=PASSPHRASE,
        max_retries=max_retries,
    )


def _raw_fill() -> dict[str, str]:
    return {
        "instType": "SPOT",
        "instId": "BTC-USDT",
        "tradeId": "trade-1",
        "ordId": "order-1",
        "side": "buy",
        "fillPx": "100",
        "fillSz": "2",
        "fee": "-0.1",
        "feeCcy": "USDT",
        "execType": "T",
        "ts": "1771200000000",
    }


def _raw_bill() -> dict[str, str]:
    return {
        "billId": "bill-1",
        "ccy": "USDT",
        "balChg": "-0.1",
        "bal": "999.9",
        "type": "2",
        "subType": "1",
        "ts": "1771200000000",
    }


def _ok_header(suffix: str) -> str:
    return "-".join(["OK", "ACCESS", suffix])
