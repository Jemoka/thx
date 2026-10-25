"""Legacy migration preserves events and publishes only complete imports."""
import runpy
import subprocess
from pathlib import Path
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from flax import serialization

pytest.importorskip("deltalake")
from deltalake import DeltaTable, write_deltalake

from theseus.base import Node
from theseus.store import ObjectReader, RecordStore, _METADATA_TYPES
from tests.test_store_clickhouse import clickhouse_server

ARROW_TYPES = {"Bool": pa.bool_(), "Int64": pa.int64(), "Float64": pa.float64(), "String": pa.string()}

MIGRATE = Path(__file__).resolve().parents[1] / "scripts" / "values_migrate_v1_to_v2.py"


@pytest.fixture
def legacy(tmp_path):
    root = tmp_path / "legacy"
    values = root / "objects" / "values"
    parent = Node(name="train", nonce="abcdef", seq=1)
    child = parent.next()
    blob = root / "objects" / "blobs" / parent.serialize()
    blob.mkdir(parents=True)
    (blob / "nodespec.json").write_text(parent.model_dump_json())
    (blob / "sample.msgpack").write_bytes(serialization.to_bytes({"sample": 7}))
    rows = []
    for index, node in enumerate((parent, parent, parent, child)):
        rows.append({
            "_x_name": node.name, "_x_nonce": node.nonce, "_x_seq": node.seq,
            "_x_parent": node.parent, "_x_event": f"event-{index}",
            "_x_write": index + 1, "_x_timestamp": index + 1,
            "_x_prc_idx": 1 if index == 2 else 0,
            "_x_execution": "execution", "_x_tag": "train",
            "_x_checkpoint": True,
            "_x_record": True if index == 0 else None,
            "_x_finished": True if index == 3 else None,
            "_x_blob": parent.serialize() if index == 0 else None,
            "loss": (3.0, 2.0, 99.0, 1.0)[index],
            "count": 2**53 + 1, "label`雪": "a'b\\c", "flag": True,
            "all_null": None,
        })
    schema = pa.schema({name: ARROW_TYPES[kind] for name, kind in _METADATA_TYPES.items()} | {
        "loss": pa.float64(), "count": pa.int64(), "label`雪": pa.string(),
        "flag": pa.bool_(), "all_null": pa.string(),
    })
    write_deltalake(values, pa.Table.from_pylist(rows[:3], schema=schema))
    fourth = pa.Table.from_pylist(rows[3:], schema=schema).append_column("new_metric", pa.array([5]))
    write_deltalake(values, fourth, mode="append", schema_mode="merge")
    # An uncommitted Parquet file is not part of the migration snapshot.
    pq.write_table(pa.table({"garbage": [True]}), values / "orphan.parquet")
    return root, parent, child, blob


@pytest.mark.parametrize("backend", ["embedded", "server"])
def test_executable_migrates_exact_events_and_records(legacy, tmp_path, request, backend):
    root, parent, child, blob = legacy
    metadata = str(tmp_path / "destination.chdb") if backend == "embedded" else request.getfixturevalue("clickhouse_server")
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert not (root / ".id").exists()
    result = subprocess.run(
        [str(MIGRATE), str(root), metadata, "--batch-size", "2"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Migration complete" in result.stdout
    assert "4/4" in result.stdout
    assert "Migrated 4 rows" in result.stdout
    identity = (root / ".id").read_text()
    assert UUID(identity).hex == identity
    expected = sorted(DeltaTable(root / "objects" / "values").to_pyarrow_table().to_pylist(), key=lambda row: row["_x_event"])
    reader = ObjectReader.local(root, metadata=metadata)
    try:
        assert reader._table == f"values_{identity}"
        actual = reader._query_arrow(f"SELECT * FROM {reader._table} ORDER BY _x_event").to_pylist()
        assert actual == expected
        assert reader.query().node(parent).select(keys=["loss"] ) == [{"loss": 2.0}]
        assert reader.parents(child) == [parent]
        assert reader.query().execution("execution").finished().checkpoint().all() == [parent, Node.deserialize(child.serialize())]
        assert reader._client.query("SHOW TABLES").result_rows == [(reader._table,)]
    finally:
        reader.close()
    records = RecordStore.local(root, metadata=metadata)
    try:
        assert records.get_record(parent)["payload"] == [{"sample": 7}]
        assert records.get_record(parent)["blob"] == blob
    finally:
        records.close()
    assert {p: (root / p).read_bytes() for p in before} == before
    repeat = subprocess.run([str(MIGRATE), str(root), metadata], capture_output=True, text=True, timeout=60)
    assert repeat.returncode != 0
    assert "already exists" in repeat.stderr
    assert (root / ".id").read_text() == identity


@pytest.mark.parametrize("failure", ["insert", "verification", "interrupt"])
def test_failed_import_retains_staging(legacy, tmp_path, monkeypatch, capsys, failure):
    migration = runpy.run_path(str(MIGRATE))
    root, *_ = legacy
    metadata = str(tmp_path / "destination.chdb")
    connect = migration["ObjectReader"].start
    calls = 0

    def fail_second_insert(self):
        connect(self)
        insert = self._client.insert_arrow
        command = self._client.command

        def incorrect_count(sql, *args, **kwargs):
            if sql.startswith("SELECT count()"):
                return 3
            return command(sql, *args, **kwargs)

        def insert_or_fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2 and failure != "verification":
                if failure == "interrupt":
                    raise KeyboardInterrupt("injected interruption")
                raise OSError("injected write failure")
            return insert(*args, **kwargs)

        monkeypatch.setattr(self._client, "insert_arrow", insert_or_fail)
        if failure == "verification":
            monkeypatch.setattr(self._client, "command", incorrect_count)

    with monkeypatch.context() as patch:
        patch.setattr(ObjectReader, "start", fail_second_insert)
        error = {"insert": OSError, "verification": RuntimeError, "interrupt": KeyboardInterrupt}[failure]
        with pytest.raises(error):
            migration["migrate"](root, metadata, batch_size=1)
    reader = ObjectReader.local(root, metadata=metadata)
    try:
        assert reader.query().all() == []
        tables = reader._client.query("SHOW TABLES").result_rows
        assert len(tables) == 1
        staging = tables[0][0]
        assert staging.startswith(f"{reader._table}_migration_")
        expected_count = 4 if failure == "verification" else 1
        assert reader._client.command(f"SELECT count() FROM {staging}") == expected_count
        output = capsys.readouterr().out
        assert staging in output
        assert "No tables were deleted" in output
    finally:
        reader.close()
    assert migration["migrate"](root, metadata, batch_size=1) == 4
    reader = ObjectReader.local(root, metadata=metadata)
    try:
        assert reader._client.command(f"SELECT count() FROM {staging}") == expected_count
        assert reader._client.command(f"SELECT count() FROM {reader._table}") == 4
    finally:
        reader.close()


def test_missing_source_does_not_create_identity_or_destination(tmp_path):
    root = tmp_path / "missing"
    root.mkdir()
    destination = tmp_path / "new.chdb"
    result = subprocess.run([str(MIGRATE), str(root), str(destination)], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert not (root / ".id").exists()
    assert not destination.exists()

def test_empty_snapshot_preserves_declared_schema(tmp_path):
    root = tmp_path / "empty"
    values = root / "objects" / "values"
    schema = pa.schema({name: ARROW_TYPES[kind] for name, kind in _METADATA_TYPES.items()} | {"unused_metric": pa.float64()})
    write_deltalake(values, pa.Table.from_pylist([], schema=schema))
    metadata = str(tmp_path / "empty.chdb")
    migration = runpy.run_path(str(MIGRATE))
    assert migration["migrate"](root, metadata) == 0
    reader = ObjectReader.local(root, metadata=metadata)
    try:
        assert reader.query().all() == []
        assert reader._schema()["unused_metric"] == "Float64"
    finally:
        reader.close()


def test_publish_refuses_concurrently_created_destination(legacy, clickhouse_server, monkeypatch):
    root, *_ = legacy
    migration = runpy.run_path(str(MIGRATE))
    connect = ObjectReader.start

    def reserve_before_publish(self):
        connect(self)
        command = self._client.command

        def race(sql, *args, **kwargs):
            if sql.startswith("RENAME TABLE"):
                staging = sql.split()[2]
                command(f"CREATE TABLE {self._table} AS {staging}")
            return command(sql, *args, **kwargs)

        monkeypatch.setattr(self._client, "command", race)

    with monkeypatch.context() as patch:
        patch.setattr(ObjectReader, "start", reserve_before_publish)
        with pytest.raises(Exception, match="already exists"):
            migration["migrate"](root, clickhouse_server)
    reader = ObjectReader.local(root, metadata=clickhouse_server)
    try:
        tables = [row[0] for row in reader._client.query("SHOW TABLES").result_rows]
        assert reader._table in tables
        staging = [table for table in tables if table.startswith(f"{reader._table}_migration_")]
        assert len(staging) == 1
        assert reader._client.command(f"SELECT count() FROM {staging[0]}") == 4
        assert reader.query().all() == []
    finally:
        reader.close()
