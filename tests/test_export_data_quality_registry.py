import json
import zipfile
from pathlib import Path

import quant_lab.export.daily as daily_export_module
from quant_lab.export.daily import export_daily_pack
from tests.test_daily_export import _fixture_lake


def test_export_data_quality_includes_registry_quality(tmp_path):
    lake_root = _fixture_lake(tmp_path)

    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(Path(result.zip_path)) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    assert "dataset_governance" in data_quality
    assert "registry_quality" in data_quality
    assert data_quality["registry_quality"]["check_count"] >= 0


def test_export_data_quality_registry_degrades_without_failing_export(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("registry failed with token=secret-value")

    monkeypatch.setattr(daily_export_module, "run_data_quality", boom)

    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(Path(result.zip_path)) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    registry_quality = data_quality["registry_quality"]
    assert registry_quality["status"] == "degraded"
    assert "secret-value" not in registry_quality["error_message"]
    assert "<REDACTED>" in registry_quality["error_message"]
