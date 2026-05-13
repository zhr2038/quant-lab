from __future__ import annotations

DEFAULT_QUOTE_ASSETS = (
    "USDT",
    "USDC",
    "USD",
    "BTC",
    "ETH",
    "EUR",
    "TRY",
)


def normalize_symbol(value: object, quote_assets: tuple[str, ...] = DEFAULT_QUOTE_ASSETS) -> str:
    """Normalize exchange and strategy symbols to the OKX-style BASE-QUOTE form."""

    if value is None:
        return ""

    text = str(value).strip().upper()
    if not text:
        return ""

    if ":" in text:
        text = text.rsplit(":", 1)[-1]

    text = text.replace("/", "-").replace("_", "-").replace(" ", "")
    while "--" in text:
        text = text.replace("--", "-")
    text = text.strip("-")
    if not text:
        return ""

    if "-" in text:
        parts = [part for part in text.split("-") if part]
        return "-".join(parts)

    for quote in sorted(quote_assets, key=len, reverse=True):
        if text.endswith(quote) and len(text) > len(quote):
            return f"{text[: -len(quote)]}-{quote}"

    return text


def normalize_optional_symbol(value: object) -> str | None:
    normalized = normalize_symbol(value)
    return normalized or None
