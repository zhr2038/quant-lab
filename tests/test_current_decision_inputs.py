from datetime import UTC, datetime, timedelta

import httpx
import pytest

from quant_lab.decision.contracts import InputSnapshot
from quant_lab.decision.current_cost_contracts import CurrentCostObservation, FeeRate
from quant_lab.decision.current_inputs import (
    BOOK_LIFETIME,
    collect_current_inputs,
    current_cost,
    merge_closed_candles,
    parse_fee,
)
from quant_lab.decision.engine import build_advice
from quant_lab.decision.storage import input_identity
from quant_lab.ingest.okx_readonly_private import (
    OKXReadOnlyClient,
    OKXReadOnlyConfig,
    OKXReadOnlySafetyError,
)
from tests.test_decision_engine import inputs, series

NOW = datetime(2026, 9, 5, 12, 10, tzinfo=UTC)
INSTRUMENT = {"groupId": "12", "minSz": "0.00001", "state": "live"}


def fee(now=NOW, taker=10):
    return FeeRate(
        symbol="BTCUSDT",
        group_id="12",
        maker_bps=8,
        taker_bps=taker,
        fetched_at=now,
        exchange_at=now,
    )


def book(now=NOW):
    return {
        "ts": str(int(now.timestamp() * 1000)),
        "asks": [["100.01", "0.25"], ["100.02", "10"]],
        "bids": [["99.99", "0.25"], ["99.98", "10"]],
    }


def estimate(now=NOW, **kw):
    defaults = dict(
        fee=fee(now),
        book=book(now),
        instrument=INSTRUMENT,
        now=now,
        anchor=None,
        notional_usdt=20,
        warnings=[],
    )
    defaults.update(kw)
    return current_cost("BTCUSDT", **defaults)


def test_current_cost_charges_fee_and_midpoint_deviation_once_for_each_size():
    cost = estimate()
    assert cost.roundtrip_bps == pytest.approx(28)
    assert [s.notional_usdt for s in cost.current.sizes] == [20, 50, 100]
    assert [s.book_roundtrip_bps for s in cost.current.sizes] == pytest.approx([2, 3, 3.5])
    assert [s.roundtrip_bps for s in cost.current.sizes] == pytest.approx([28, 29, 29.5])
    assert cost.current.valid_until == NOW + BOOK_LIFETIME
    assert not cost.trusted_for_paper and cost.actual_sample_count == 0
    assert cost.current.calibrated is False


@pytest.mark.parametrize("taker, expected", [(0, 8), (-2, 8), (10, 28)])
def test_zero_fee_is_valid_and_rebate_is_preserved_without_optimistic_credit(taker, expected):
    cost = estimate(fee=fee(taker=taker))
    assert cost.roundtrip_bps == pytest.approx(expected)
    assert cost.current.fee.taker_bps == taker
    if taker < 0:
        assert "REBATE_NOT_CREDITED_IN_ESTIMATE" in cost.missing_reasons


def test_depth_failure_does_not_hide_size_dependence_or_invent_cost():
    shallow = book()
    shallow["asks"] = shallow["asks"][:1]
    shallow["bids"] = shallow["bids"][:1]
    cost = estimate(book=shallow)
    assert cost.roundtrip_bps == pytest.approx(28)
    assert all(
        s.status == "INSUFFICIENT_DEPTH" and s.roundtrip_bps is None for s in cost.current.sizes[1:]
    )
    cost = estimate(book=shallow, notional_usdt=50)
    assert cost.roundtrip_bps is None
    assert cost.current.status == "UNAVAILABLE"


def test_minimum_size_is_explicit():
    cost = estimate(instrument={**INSTRUMENT, "minSz": "0.3"})
    assert cost.roundtrip_bps is None
    assert "BELOW_MINIMUM_SIZE" in cost.missing_reasons


@pytest.mark.parametrize(
    "change",
    [
        {"ts": str(int((NOW - timedelta(seconds=61)).timestamp() * 1000))},
        {"ts": str(int((NOW + timedelta(seconds=1)).timestamp() * 1000))},
        {"asks": [["99", "1"]]},
        {"asks": [["100.02", "1"], ["100.01", "1"]]},
        {"asks": [["nan", "1"]]},
        {"bids": []},
    ],
)
def test_invalid_book_is_unknown_not_a_zero_cost(change):
    cost = estimate(book={**book(), **change})
    assert cost.roundtrip_bps is None
    assert cost.current.status == "UNAVAILABLE"
    assert "CURRENT_BOOK_OR_INSTRUMENT_UNAVAILABLE" in cost.missing_reasons


def test_missing_and_expired_account_fee_do_not_use_global_default():
    for value in (None, fee(NOW - timedelta(hours=25))):
        cost = estimate(fee=value)
        assert cost.roundtrip_bps is None
        assert "ACCOUNT_FEE_UNAVAILABLE" in cost.missing_reasons


def raw_fee(now=NOW):
    return [
        {
            "instType": "SPOT",
            "ts": str(int(now.timestamp() * 1000)),
            "feeGroup": [
                {"groupId": "11", "maker": "-0.003", "taker": "-0.004"},
                {"groupId": "12", "maker": "0.0001", "taker": "0"},
            ],
            "maker": "-0.009",
            "taker": "-0.01",
        }
    ]


def test_fee_group_binding_uses_new_fields_and_zero_not_legacy_default():
    observed = parse_fee("BTCUSDT", raw_fee(), "12", NOW)
    assert observed.taker_bps == 0
    assert observed.maker_bps == -1
    with pytest.raises(ValueError, match="uniquely"):
        parse_fee("BTCUSDT", raw_fee(), "missing", NOW)
    with pytest.raises(ValueError, match="timestamp"):
        parse_fee("BTCUSDT", raw_fee(NOW - timedelta(hours=2)), "12", NOW)


def test_fee_endpoint_is_symbol_bound_and_get_only():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"code": "0", "data": raw_fee()})

    with httpx.Client(
        base_url="https://example.test", transport=httpx.MockTransport(respond)
    ) as http:
        client = OKXReadOnlyClient(
            OKXReadOnlyConfig(
                api_key="test",
                secret_key="test",
                passphrase="test",
                max_retries=0,
            ),
            http,
        )
        assert client.get_spot_fee_rates("BTC-USDT") == raw_fee()
        assert calls[0].method == "GET"
        assert calls[0].url.path == "/api/v5/account/trade-fee"
        assert dict(calls[0].url.params) == {"instType": "SPOT", "instId": "BTC-USDT"}
        with pytest.raises(OKXReadOnlySafetyError):
            client._private_get("/api/v5/account/trade-fee", {}, method="POST")
        assert len(calls) == 1


class Public:
    def get_instruments(self, inst_type):
        assert inst_type == "SPOT"
        return [
            {**INSTRUMENT, "instId": s} for s in ("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT")
        ]

    def get_candles(self, inst, bar, limit):
        assert bar == "1H" and limit == 200
        return [
            {
                "ts": str(int(NOW.replace(minute=0).timestamp() * 1000)),
                "o": "100",
                "h": "100",
                "l": "100",
                "c": "100",
                "vol": "1",
                "confirm": "0",
            }
        ]

    def get_orderbook(self, inst, sz):
        assert sz == 20
        return book()


class Private:
    permission = "read_only"

    def __init__(self):
        self.reads = []

    def get_account_config(self):
        self.reads.append("permissions")
        return {"perm": self.permission}

    def get_spot_fee_rates(self, inst):
        self.reads.append(inst)
        return raw_fee()


def snapshot(collected):
    value = InputSnapshot(
        snapshot_id="input-" + "0" * 64,
        producer_commit="abcdef1",
        generated_at=collected.generated_at,
        bars=collected.bars,
        costs=collected.costs,
        warnings=collected.warnings,
        signature="test",
    )
    return value.model_copy(update={"snapshot_id": input_identity(value)})


def test_collector_has_no_lake_dependency_and_cache_never_restamps_fee_time():
    private = Private()
    first = collect_current_inputs(None, clock=lambda: NOW, public=Public(), private=private)
    assert len(first.costs) == 4 and not first.bars  # Open candle never enters the research.
    assert len(private.reads) == 5
    previous = snapshot(first)
    again = collect_current_inputs(
        previous, clock=lambda: NOW + timedelta(seconds=30), public=Public(), private=private
    )
    assert len(private.reads) == 5
    assert all(c.current.fee.fetched_at == NOW for c in again.costs)
    assert all(c.as_of == NOW for c in again.costs)
    assert snapshot(again).snapshot_id == previous.snapshot_id


def test_key_with_trade_permission_is_rejected_without_fee_requests():
    private = Private()
    private.permission = "read_only,trade"
    result = collect_current_inputs(None, clock=lambda: NOW, public=Public(), private=private)
    assert private.reads == ["permissions"]
    assert all(c.roundtrip_bps is None for c in result.costs)
    assert all("READONLY_FEE_ACCESS_UNAVAILABLE" in c.missing_reasons for c in result.costs)


def test_fee_refresh_failure_keeps_original_time_but_expiry_still_blocks_cost():
    private = Private()
    prior = snapshot(
        collect_current_inputs(None, clock=lambda: NOW, public=Public(), private=private)
    )
    private.permission = ""
    result = collect_current_inputs(
        prior, clock=lambda: NOW + timedelta(hours=25), public=Public(), private=private
    )
    assert all(c.roundtrip_bps is None for c in result.costs)
    assert all("ACCOUNT_FEE_EXPIRED" in c.missing_reasons for c in result.costs)


def test_closed_bar_is_added_immediately_and_re_read_does_not_change_ingest_time():
    row = {
        "ts": str(int((NOW.replace(minute=0) - timedelta(hours=1)).timestamp() * 1000)),
        "o": "100",
        "h": "101",
        "l": "99",
        "c": "100.5",
        "vol": "1",
        "confirm": "1",
    }
    first = merge_closed_candles("BTCUSDT", [row], [], NOW)
    assert len(first) == 1
    assert merge_closed_candles("BTCUSDT", [row], first, NOW + timedelta(minutes=5)) == first
    corrected = merge_closed_candles(
        "BTCUSDT", [{**row, "c": "100.6"}], first, NOW + timedelta(minutes=5)
    )
    assert corrected[0].ingest_ts == NOW + timedelta(minutes=5)
    assert first[0].close == 100.5


def test_old_signed_shapes_and_new_cost_details_roundtrip_without_identity_changes():
    bars = series()
    old = inputs(bars, bars[-1].ingest_ts)
    raw = old.model_dump(mode="json")
    assert "current" not in raw["costs"][0]
    assert InputSnapshot.model_validate(raw).model_dump(mode="json") == raw
    new = snapshot(
        collect_current_inputs(None, clock=lambda: NOW, public=Public(), private=Private())
    )
    assert isinstance(new.costs[0], CurrentCostObservation)
    reread = InputSnapshot.model_validate_json(new.model_dump_json())
    assert isinstance(reread.costs[0], CurrentCostObservation)
    assert input_identity(reread) == new.snapshot_id


def test_current_cost_expiry_caps_advice_and_no_calibration_is_invented():
    bars = series(rate=0.002)
    now = bars[-1].ingest_ts + timedelta(minutes=5)
    inp = inputs(bars, now).model_copy(update={"costs": [estimate(now)]})
    advice = build_advice(bars, inputs=inp, symbol="BTCUSDT", horizon=4, now=now)
    assert advice.action == "KEEP_BASELINE"
    assert "COST_REQUIRES_CALIBRATION" in advice.reason_codes
    assert advice.expires_at == now + BOOK_LIFETIME
    expired = build_advice(bars, inputs=inp, symbol="BTCUSDT", horizon=4, now=now + BOOK_LIFETIME)
    assert expired.action == "NO_VIEW"
    assert "CURRENT_COST_EXPIRED_OR_UNAVAILABLE" in expired.reason_codes
