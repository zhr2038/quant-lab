import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

import quant_lab.export.daily as daily_export_module
from quant_lab.data.lake import write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.ops.data_quality import run_data_quality
from quant_lab.ops.dataset_registry import dataset_registry
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


def test_registry_quality_uses_export_frame_over_stale_disk_copy(tmp_path):
    now = datetime(2026, 7, 17, 8, 0, tzinfo=UTC)
    spec = dataset_registry()["paper_runtime_freshness"]
    stale = pl.DataFrame(
        [
            {
                "check_name": f"check-{index}",
                "status": "PASS",
                "generated_at": (now - timedelta(hours=4)).isoformat(),
            }
            for index in range(5)
        ]
    )
    fresh = stale.with_columns(pl.lit(now.isoformat()).alias("generated_at"))
    write_parquet_dataset(stale, tmp_path / spec.relative_path)

    result = run_data_quality(
        tmp_path,
        dataset_names=[spec.dataset_id],
        reference_at=now,
        frame_overrides={spec.dataset_id: fresh},
    )

    freshness = next(check for check in result.checks if check.rule == "freshness")
    assert freshness.status == "PASS"
    assert freshness.observed_value == "0"
