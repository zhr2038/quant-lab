from types import SimpleNamespace

from quant_lab.decision.api import reference_contract_view


def test_legacy_display_does_not_invent_strategy_or_change_signed_payload():
    advice = SimpleNamespace(advice_id="old", experiment_version="old-experiment",
                             cost=SimpleNamespace(version="old-cost"), horizon_hours=24)
    result = SimpleNamespace(schema_version="qlab.decision.result.v1", worker_commit="old-worker",
                             advice=[advice])
    before = dict(advice.__dict__)
    view = reference_contract_view(result)
    assert view[0]["strategy_version"] is None
    assert view[0]["reference_schema"] == "qlab.decision.result.v1"
    assert advice.__dict__ == before
