import random

from quant_lab.research.factor_research.overfit import compute_overfit_diagnostics


def test_overfit_diagnostics_are_inconclusive_without_enough_variants() -> None:
    result = compute_overfit_diagnostics({"only": [0.01] * 40})

    assert result.status == "INCONCLUSIVE_OVERFIT_DIAGNOSTICS"
    assert result.pbo is None
    assert "at_least_two_variants_required_for_pbo" in result.blockers


def test_overfit_diagnostics_report_pbo_dsr_and_selection_degradation() -> None:
    randomizer = random.Random(31)
    variants = {
        "robust": [0.025 + randomizer.gauss(0.0, 0.01) for _ in range(96)],
        "weak": [0.002 + randomizer.gauss(0.0, 0.02) for _ in range(96)],
        "negative": [-0.005 + randomizer.gauss(0.0, 0.02) for _ in range(96)],
    }

    result = compute_overfit_diagnostics(variants)

    assert result.selected_variant_id == "robust"
    assert result.pbo is not None
    assert result.dsr_probability is not None
    assert result.selection_degradation is not None
    assert result.cscv_split_count > 0
