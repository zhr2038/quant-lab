from pathlib import Path

POINT_IN_TIME_COST_EVALUATION_DAYS = 30
MAX_POINT_IN_TIME_COST_AGE_HOURS = 36
MIN_DATA_COVERAGE = 0.80
MIN_RESEARCH_PROXY_ROUNDTRIP_COST_BPS = 30.0
MIN_POINT_IN_TIME_SPREAD_SAMPLES_PER_DAY = 60

ORDERBOOK_SPREAD_1M_DATASET = Path("silver") / "orderbook_spread_1m"

RESEARCH_COST_SOURCES = frozenset(
    {
        "actual",
        "actual_fills",
        "actual_okx_fills_and_bills",
        "actual_okx_fills_fee_missing",
        "mixed_actual_proxy",
        "public_proxy",
        "public_spread_proxy",
        "bootstrap_cost_probe",
    }
)
TRUSTED_COST_SOURCES = frozenset(
    {
        "actual",
        "actual_fills",
        "actual_okx_fills_and_bills",
        "actual_okx_fills_fee_missing",
        "mixed_actual_proxy",
    }
)
PUBLIC_PROXY_COST_SOURCES = frozenset({"public_proxy", "public_spread_proxy"})
