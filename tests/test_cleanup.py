"""Value-table maintenance preserves committed rows and reader snapshots."""

import os
import importlib
import shutil
from time import time
from unittest.mock import Mock

import pyarrow as pa
import pytest
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import CommitFailedError
from deltalake.table import TableOptimizer
from typer.testing import CliRunner

from theseus.cli.app import app
from theseus.store import ObjectStore
import theseus.store as store_module


def seed(values):
    for value in range(3):
        write_deltalake(values, pa.table({"value": [value]}), mode="append")


@pytest.mark.parametrize("expire", [False, True])
def test_cleanup_compacts_and_vacuums_with_selected_retention(tmp_path, monkeypatch, expire):
    values = tmp_path / "objects" / "values"
    seed(values)
    snapshot = DeltaTable(values)
    old_files = snapshot.file_uris()
    expired = values / "orphan.parquet"
    shutil.copyfile(old_files[0], expired)
    old_time = time() - 8 * 24 * 3600
    os.utime(expired, (old_time, old_time))
    recent = values / "recent-orphan.parquet"
    shutil.copyfile(old_files[0], recent)
    cleanup_module = importlib.import_module("theseus.cli.cleanup")
    delay = Mock()
    monkeypatch.setattr(cleanup_module, "sleep", delay)

    result = CliRunner().invoke(app, ["cleanup", str(tmp_path), *(["--expire"] if expire else [])])

    assert result.exit_code == 0, result.output
    assert "Value table cleanup" in result.output
    assert "CLEANUP | compacting" in result.output
    assert "CLEANUP | vacuuming" in result.output
    assert not expired.exists()
    assert recent.exists() is not expire
    assert len(DeltaTable(values).file_uris()) == 1
    if expire:
        delay.assert_called_once_with(10)
        assert "PERMANENTLY DELETES" in result.output
        assert all(not os.path.exists(path) for path in old_files)
    else:
        delay.assert_not_called()
        # Default retention preserves readers bound before compaction.
        assert sorted(snapshot.to_pyarrow_table()["value"].to_pylist()) == [0, 1, 2]
    assert sorted(DeltaTable(values).to_pyarrow_table()["value"].to_pylist()) == [0, 1, 2]


def test_cleanup_expire_can_be_cancelled_before_mutation(tmp_path, monkeypatch):
    cleanup_module = importlib.import_module("theseus.cli.cleanup")
    delay = Mock(side_effect=KeyboardInterrupt)
    compact = Mock()
    vacuum = Mock()
    monkeypatch.setattr(cleanup_module, "sleep", delay)
    monkeypatch.setattr(ObjectStore, "compact", compact)
    monkeypatch.setattr(DeltaTable, "vacuum", vacuum)

    result = CliRunner().invoke(app, ["cleanup", str(tmp_path), "--expire"])

    assert result.exit_code != 0
    assert "Press Ctrl-C within 10 seconds" in result.output
    delay.assert_called_once_with(10)
    compact.assert_not_called()
    vacuum.assert_not_called()


def test_cleanup_missing_table_is_noop(tmp_path):
    result = CliRunner().invoke(app, ["cleanup", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "no Delta value table" in result.output
    assert not (tmp_path / "objects").exists()


@pytest.mark.parametrize("schema_change", [False, True])
def test_compaction_preserves_concurrent_append_and_retries_schema_change(
    tmp_path, monkeypatch, schema_change,
):
    values = tmp_path / "objects" / "values"
    seed(values)
    original = TableOptimizer.compact
    calls = 0

    def racing_compact(optimizer):
        nonlocal calls
        calls += 1
        if calls == 1:
            data = {"value": [3]}
            if schema_change:
                data["new_column"] = [4]
            write_deltalake(values, pa.table(data), mode="append", schema_mode="merge")
        return original(optimizer)

    monkeypatch.setattr(TableOptimizer, "compact", racing_compact)
    monkeypatch.setattr(store_module, "_DELTA_IO_RETRY_SECONDS", 0)
    ObjectStore.compact(tmp_path)

    assert calls == (2 if schema_change else 1)
    rows = DeltaTable(values).to_pyarrow_table()
    assert sorted(rows["value"].to_pylist()) == [0, 1, 2, 3]
    if schema_change:
        assert rows.filter(pa.compute.equal(rows["value"], 3))["new_column"].to_pylist() == [4]


def test_close_logs_compaction_failure_after_flushing(tmp_path, monkeypatch):
    from theseus.base import Node

    store = ObjectStore.local(tmp_path)
    store.value(Node(name="test"), {"value": 1})
    warning = Mock()
    monkeypatch.setattr(store_module.logger, "warning", warning)
    monkeypatch.setattr(
        ObjectStore, "compact", Mock(side_effect=CommitFailedError("conflict")),
    )
    store.close()
    warning.assert_called_once()
    assert store.query().select() == [{"value": 1}]


def test_cleanup_stops_before_vacuum_after_repeated_conflicts(tmp_path, monkeypatch):
    seed(tmp_path / "objects" / "values")
    compact = Mock(side_effect=CommitFailedError("concurrent schema change"))
    vacuum = Mock()
    monkeypatch.setattr(TableOptimizer, "compact", compact)
    monkeypatch.setattr(DeltaTable, "vacuum", vacuum)
    monkeypatch.setattr(store_module, "_DELTA_IO_RETRY_SECONDS", 0)

    result = CliRunner().invoke(app, ["cleanup", str(tmp_path)])

    assert result.exit_code == 1
    assert "concurrent schema change" in result.output
    assert compact.call_count == store_module._DELTA_IO_ATTEMPTS
    vacuum.assert_not_called()
