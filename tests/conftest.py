import pytest


@pytest.fixture(autouse=True)
def isolate_default_lake_root(monkeypatch, tmp_path):
    """Keep tests from accidentally writing to the production default lake path."""

    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path / "lake"))
