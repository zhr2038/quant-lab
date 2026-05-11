# OKX Read-Only Private Collector

The OKX read-only private collector is optional. It is used only to read fills,
bills, order history, and account configuration for cost-model calibration and
research audit. It must never mutate exchange or strategy state.

## Credential Requirements

Environment variables:

```bash
export OKX_API_KEY=...
export OKX_SECRET_KEY=...
export OKX_PASSPHRASE=...
```

The OKX key must have **Read** permission only.

Forbidden permissions:

- Trade
- Withdraw

Do not commit credentials, print them to logs, put them in config files, or write
them to lake/archive/export files.

## Allowed Endpoints

Only private GET requests are allowed:

- `GET /api/v5/trade/fills-history`
- `GET /api/v5/trade/orders-history`
- `GET /api/v5/trade/orders-history-archive`
- `GET /api/v5/account/bills`
- `GET /api/v5/account/bills-archive`
- `GET /api/v5/account/config`

All non-GET methods and non-allowlisted private endpoints are rejected before an
HTTP request is sent.

## Lake Outputs

Bronze datasets contain sanitized raw payloads:

- `bronze/okx_private_readonly/fills_history`
- `bronze/okx_private_readonly/bills`
- `bronze/okx_private_readonly/orders_history`

Silver datasets contain normalized records:

- `silver/fill_event`
- `silver/account_bill`
- `silver/order_event`

Sensitive fields whose names look like key, secret, passphrase, or signature are
removed from bronze payloads before writing. Silver schemas do not contain
credential fields.

## Backfill

```bash
qlab okx-backfill-readonly \
  --inst-type SPOT \
  --inst-id BTC-USDT \
  --ccy USDT \
  --lake-root /var/lib/quant-lab/lake \
  --max-pages 2
```

The command fetches fills, account bills, and order history via GET and writes
bronze/silver lake datasets. Tests mock HTTP responses and never access real
OKX.
