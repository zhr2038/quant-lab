from datetime import UTC, datetime, timedelta

from quant_lab.ops.retention import prune_quant_lab_storage


def test_retention_dry_run_reports_without_deleting_files(tmp_path):
    base = tmp_path / "quant-lab"
    export_root = base / "exports"
    export_root.mkdir(parents=True)
    packs = []
    for index in range(7):
        pack = export_root / f"quant_lab_expert_pack_{index}.zip"
        pack.write_text("pack", encoding="utf-8")
        mtime = (datetime.now(UTC) - timedelta(minutes=index)).timestamp()
        pack.touch()
        packs.append((pack, mtime))

    result = prune_quant_lab_storage(base, keep_export_packs=5, dry_run=True)

    assert result.export_removed_files == 2
    assert all(pack.exists() for pack, _mtime in packs)
    assert result.to_dict()["dry_run"] is True
