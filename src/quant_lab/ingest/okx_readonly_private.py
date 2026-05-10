import base64
import hashlib
import hmac
import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from quant_lab.contracts.models import AccountBill, FillEvent
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset

OKX_READONLY_PRIVATE_SOURCE = "okx_readonly_private"

_TRADE_PREFIX = "/api/v5/trade"
_ACCOUNT_PREFIX = "/api/v5/account"
FILLS_HISTORY_ENDPOINT = f"{_TRADE_PREFIX}/fills-history"
ORDERS_HISTORY_ENDPOINT = f"{_TRADE_PREFIX}/orders-history"
ORDERS_HISTORY_ARCHIVE_ENDPOINT = f"{_TRADE_PREFIX}/orders-history-archive"
ACCOUNT_BILLS_ENDPOINT = f"{_ACCOUNT_PREFIX}/bills"
ACCOUNT_BILLS_ARCHIVE_ENDPOINT = f"{_ACCOUNT_PREFIX}/bills-archive"
ACCOUNT_CONFIG_ENDPOINT = f"{_ACCOUNT_PREFIX}/config"

ALLOWED_GET_ENDPOINTS = frozenset(
    {
        FILLS_HISTORY_ENDPOINT,
        ORDERS_HISTORY_ENDPOINT,
        ORDERS_HISTORY_ARCHIVE_ENDPOINT,
        ACCOUNT_BILLS_ENDPOINT,
        ACCOUNT_BILLS_ARCHIVE_ENDPOINT,
        ACCOUNT_CONFIG_ENDPOINT,
    }
)

BRONZE_FILLS_DATASET = Path("bronze") / "okx_private_readonly" / "fills_history"
BRONZE_BILLS_DATASET = Path("bronze") / "okx_private_readonly" / "bills"
SILVER_FILL_EVENT_DATASET = Path("silver") / "fill_event"
SILVER_ACCOUNT_BILL_DATASET = Path("silver") / "account_bill"

BRONZE_PRIVATE_SCHEMA = {
    "endpoint": pl.Utf8,
    "ingest_ts": pl.Utf8,
    "raw_json": pl.Utf8,
}

FILL_EVENT_SCHEMA = {
    "venue": pl.Utf8,
    "inst_type": pl.Utf8,
    "inst_id": pl.Utf8,
    "trade_id": pl.Utf8,
    "order_id": pl.Utf8,
    "side": pl.Utf8,
    "fill_price": pl.Float64,
    "fill_size": pl.Float64,
    "fee": pl.Float64,
    "fee_currency": pl.Utf8,
    "liquidity": pl.Utf8,
    "ts": pl.Utf8,
    "source": pl.Utf8,
}

ACCOUNT_BILL_SCHEMA = {
    "venue": pl.Utf8,
    "bill_id": pl.Utf8,
    "ccy": pl.Utf8,
    "amount": pl.Float64,
    "balance": pl.Float64,
    "bill_type": pl.Utf8,
    "sub_type": pl.Utf8,
    "ts": pl.Utf8,
    "source": pl.Utf8,
}


class OKXReadOnlyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = "https://www.okx.com"
    api_key: SecretStr
    secret_key: SecretStr
    passphrase: SecretStr
    timeout_seconds: float = Field(default=10.0, gt=0)
    max_retries: int = Field(default=3, ge=0)

    @classmethod
    def from_env(cls) -> "OKXReadOnlyConfig":
        required = {
            "api_key": "OKX_API_KEY",
            "secret_key": "OKX_SECRET_KEY",
            "passphrase": "OKX_PASSPHRASE",
        }
        values: dict[str, str] = {}
        missing: list[str] = []
        for field_name, env_name in required.items():
            value = os.getenv(env_name)
            if value:
                values[field_name] = value
            else:
                missing.append(env_name)
        if missing:
            missing_names = ", ".join(missing)
            raise OKXReadOnlyConfigError(
                f"Missing required OKX read-only private environment variables: {missing_names}"
            )
        return cls(**values)


class OKXReadOnlyError(RuntimeError):
    pass


class OKXReadOnlyConfigError(OKXReadOnlyError):
    pass


class OKXReadOnlySafetyError(OKXReadOnlyError):
    pass


class OKXReadOnlyTimeout(OKXReadOnlyError):
    pass


class OKXReadOnlyAPIError(OKXReadOnlyError):
    pass


class OKXReadOnlyClient:
    def __init__(
        self,
        config: OKXReadOnlyConfig,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._client = http_client or httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )

    def get_fills_history(
        self,
        inst_type: str,
        inst_id: str | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._private_get(
            FILLS_HISTORY_ENDPOINT,
            _drop_none(
                {
                    "instType": inst_type,
                    "instId": inst_id,
                    "after": after,
                    "before": before,
                    "limit": str(limit),
                }
            ),
        )

    def get_orders_history(
        self,
        inst_type: str,
        inst_id: str | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._private_get(
            ORDERS_HISTORY_ENDPOINT,
            _drop_none(
                {
                    "instType": inst_type,
                    "instId": inst_id,
                    "after": after,
                    "before": before,
                    "limit": str(limit),
                }
            ),
        )

    def get_orders_history_archive(
        self,
        inst_type: str,
        inst_id: str | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._private_get(
            ORDERS_HISTORY_ARCHIVE_ENDPOINT,
            _drop_none(
                {
                    "instType": inst_type,
                    "instId": inst_id,
                    "after": after,
                    "before": before,
                    "limit": str(limit),
                }
            ),
        )

    def get_account_bills(
        self,
        ccy: str | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._private_get(
            ACCOUNT_BILLS_ENDPOINT,
            _drop_none(
                {
                    "ccy": ccy,
                    "after": after,
                    "before": before,
                    "limit": str(limit),
                }
            ),
        )

    def get_account_bills_archive(
        self,
        ccy: str | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._private_get(
            ACCOUNT_BILLS_ARCHIVE_ENDPOINT,
            _drop_none(
                {
                    "ccy": ccy,
                    "after": after,
                    "before": before,
                    "limit": str(limit),
                }
            ),
        )

    def get_account_config(self) -> dict[str, Any]:
        data = self._private_get(ACCOUNT_CONFIG_ENDPOINT, {})
        return data[0] if data else {}

    def _private_get(
        self,
        endpoint: str,
        params: Mapping[str, str],
        method: str = "GET",
    ) -> list[dict[str, Any]]:
        payload = self._request_private(method=method, endpoint=endpoint, params=params)
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise OKXReadOnlyAPIError(f"OKX read-only response data is not a list for {endpoint}")
        return data

    def _request_private(
        self,
        method: str,
        endpoint: str,
        params: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        normalized_method = method.upper()
        if normalized_method != "GET":
            raise OKXReadOnlySafetyError("OKX read-only private collector only allows GET")
        if endpoint not in ALLOWED_GET_ENDPOINTS:
            raise OKXReadOnlySafetyError(f"Endpoint is not in OKX read-only allowlist: {endpoint}")

        query_params = dict(params or {})
        request_path = _request_path(endpoint, query_params)
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            timestamp = _okx_timestamp()
            headers = self._auth_headers(timestamp=timestamp, request_path=request_path)
            try:
                response = self._client.request(
                    "GET",
                    endpoint,
                    params=query_params,
                    headers=headers,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                    time.sleep(0)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise OKXReadOnlyAPIError("OKX read-only response is not a JSON object")
                if payload.get("code") != "0":
                    raise OKXReadOnlyAPIError(
                        "OKX read-only API error "
                        f"code={payload.get('code')} msg={self._redact(payload.get('msg'))}"
                    )
                return payload
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(0)
                    continue
                raise OKXReadOnlyTimeout(f"OKX read-only request timed out for {endpoint}") from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(0)
                    continue
                raise OKXReadOnlyError(
                    f"OKX read-only request failed for {endpoint}: {self._redact(exc)}"
                ) from exc

        raise OKXReadOnlyError(
            f"OKX read-only request failed for {endpoint}: {self._redact(last_error)}"
        )

    def _auth_headers(self, timestamp: str, request_path: str) -> dict[str, str]:
        signature = _sign_get_request(
            timestamp=timestamp,
            request_path=request_path,
            secret_key=self.config.secret_key,
        )
        return {
            _ok_access_header("KEY"): self.config.api_key.get_secret_value(),
            _ok_access_header("SIGN"): signature,
            _ok_access_header("TIMESTAMP"): timestamp,
            _ok_access_header("PASSPHRASE"): self.config.passphrase.get_secret_value(),
        }

    def _redact(self, value: Any) -> str:
        text = "" if value is None else str(value)
        for secret in [
            self.config.api_key.get_secret_value(),
            self.config.secret_key.get_secret_value(),
            self.config.passphrase.get_secret_value(),
        ]:
            if secret:
                text = text.replace(secret, "<redacted>")
        return text


def normalize_okx_fills(raw_fills: Sequence[Mapping[str, Any]]) -> list[FillEvent]:
    records: list[FillEvent] = []
    for item in raw_fills:
        records.append(
            FillEvent(
                inst_type=str(item["instType"]),
                inst_id=str(item["instId"]),
                trade_id=str(item["tradeId"]),
                order_id=str(item["ordId"]),
                side=str(item["side"]),
                fill_price=float(item["fillPx"]),
                fill_size=float(item["fillSz"]),
                fee=float(item.get("fee", 0)),
                fee_currency=str(item.get("feeCcy") or item.get("feeCurrency")),
                liquidity=_optional_string(item.get("execType") or item.get("liquidity")),
                ts=_timestamp_ms_to_utc(item["ts"]),
            )
        )
    return records


def normalize_okx_bills(raw_bills: Sequence[Mapping[str, Any]]) -> list[AccountBill]:
    records: list[AccountBill] = []
    for item in raw_bills:
        records.append(
            AccountBill(
                bill_id=str(item["billId"]),
                ccy=str(item["ccy"]),
                amount=float(item.get("balChg") or item.get("amt") or 0),
                balance=float(item.get("bal") or item.get("balance") or 0),
                bill_type=str(item["type"]),
                sub_type=str(item["subType"]),
                ts=_timestamp_ms_to_utc(item["ts"]),
            )
        )
    return records


def publish_okx_fills_to_lake(
    raw_fills: Sequence[Mapping[str, Any]],
    lake_root: str | Path,
) -> dict[str, int]:
    root = Path(lake_root)
    fill_events = normalize_okx_fills(raw_fills)
    bronze_rows = _upsert_frame(
        root / BRONZE_FILLS_DATASET,
        _bronze_private_frame(FILLS_HISTORY_ENDPOINT, raw_fills),
        key_columns=["endpoint", "raw_json"],
    )
    silver_rows = _upsert_frame(
        root / SILVER_FILL_EVENT_DATASET,
        _fill_event_frame(fill_events),
        key_columns=["inst_id", "trade_id", "order_id", "ts"],
    )
    return {"bronze_fills_rows": bronze_rows, "fill_event_rows": silver_rows}


def publish_okx_bills_to_lake(
    raw_bills: Sequence[Mapping[str, Any]],
    lake_root: str | Path,
) -> dict[str, int]:
    root = Path(lake_root)
    account_bills = normalize_okx_bills(raw_bills)
    bronze_rows = _upsert_frame(
        root / BRONZE_BILLS_DATASET,
        _bronze_private_frame(ACCOUNT_BILLS_ENDPOINT, raw_bills),
        key_columns=["endpoint", "raw_json"],
    )
    silver_rows = _upsert_frame(
        root / SILVER_ACCOUNT_BILL_DATASET,
        _account_bill_frame(account_bills),
        key_columns=["bill_id", "ccy", "ts"],
    )
    return {"bronze_bills_rows": bronze_rows, "account_bill_rows": silver_rows}


def _sign_get_request(timestamp: str, request_path: str, secret_key: SecretStr) -> str:
    pre_hash = f"{timestamp}GET{request_path}"
    digest = hmac.new(
        secret_key.get_secret_value().encode("utf-8"),
        pre_hash.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def _request_path(endpoint: str, params: Mapping[str, str]) -> str:
    if not params:
        return endpoint
    return f"{endpoint}?{urlencode(list(params.items()))}"


def _okx_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _drop_none(values: Mapping[str, str | None]) -> dict[str, str]:
    return {key: value for key, value in values.items() if value is not None}


def _ok_access_header(suffix: str) -> str:
    return "-".join(["OK", "ACCESS", suffix])


def _timestamp_ms_to_utc(value: Any) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC)


def _upsert_frame(dataset_path: Path, new_df: pl.DataFrame, key_columns: list[str]) -> int:
    existing_df = read_parquet_dataset(dataset_path)
    frames = [frame for frame in [existing_df, new_df] if not frame.is_empty()]
    combined = pl.concat(frames, how="diagonal_relaxed") if frames else new_df
    if not combined.is_empty():
        available_keys = [column for column in key_columns if column in combined.columns]
        if available_keys:
            combined = combined.unique(subset=available_keys, keep="last", maintain_order=True)
    write_parquet_dataset(combined, dataset_path)
    return combined.height


def _bronze_private_frame(endpoint: str, raw_items: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    ingest_ts = _okx_timestamp()
    rows = [
        {
            "endpoint": endpoint,
            "ingest_ts": ingest_ts,
            "raw_json": _json_dumps(_sanitize_private_payload(item)),
        }
        for item in raw_items
    ]
    return pl.DataFrame(rows, schema=BRONZE_PRIVATE_SCHEMA, orient="row")


def _fill_event_frame(records: Sequence[FillEvent]) -> pl.DataFrame:
    rows = [record.model_dump(mode="json") for record in records]
    return pl.DataFrame(rows, schema=FILL_EVENT_SCHEMA, orient="row")


def _account_bill_frame(records: Sequence[AccountBill]) -> pl.DataFrame:
    rows = [record.model_dump(mode="json") for record in records]
    return pl.DataFrame(rows, schema=ACCOUNT_BILL_SCHEMA, orient="row")


def _sanitize_private_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    sensitive_fragments = ("key", "secret", "passphrase", "sign")
    return {
        str(key): value
        for key, value in item.items()
        if not any(fragment in str(key).lower() for fragment in sensitive_fragments)
    }


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def iter_fill_event_dicts(records: Iterable[FillEvent]) -> list[dict[str, Any]]:
    return [record.model_dump(mode="json") for record in records]
