"""Delta transaction, scalar schema, and concurrent publication contracts."""

import errno
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from deltalake import DeltaTable
from deltalake.exceptions import CommitFailedError

import theseus.store as store_module
from theseus.base import Node
from theseus.store import ObjectReader, ObjectStore


def publish_values(root, writer, barrier):
    # Keep each spawned CPU worker's JAX thread pool small.
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    store = ObjectStore.local(root)
    barrier.wait(timeout=30)
    for batch in range(6):
        store.start()
        for index in range(4):
            store.value(
                Node(name=f"writer-{writer}", nonce="abcdef", seq=4 * batch + index),
                {"shared": index},
            )
        store.close()


@pytest.mark.parametrize(
    "initial,incoming,expected",
    [
        (1.0, 2, 2.0),
        (1, 2.0, 2),
        ("one", 2, "2"),
        (True, None, True),
        (1, None, 1),
        (1.0, None, 1.0),
    ],
)
def test_existing_column_accepts_only_safe_casts(tmp_path, initial, incoming, expected):
    node = Node(name="types", nonce="abcdef")
    store = ObjectStore.local(tmp_path)
    store.value(node, {"value": initial})
    store.close()
    schema = DeltaTable(store.values()).schema()
    store.start()
    store.value(node, {"value": incoming})
    store.close()
    assert DeltaTable(store.values()).schema() == schema
    assert store.query().select() == [{"value": expected}]


@pytest.mark.parametrize(
    "initial,incoming",
    [(1, 1.5), (1, "2"), (1.0, "2"), (True, 1), (1.0, 2**53 + 1), (1, 2**63)],
)
def test_unsafe_cast_fails_without_committing(tmp_path, initial, incoming):
    store = ObjectStore.local(tmp_path)
    node = Node(name="types", nonce="abcdef")
    store.value(node, {"value": initial})
    store.close()
    version = DeltaTable(store.values()).version()
    store.start()
    store.value(node, {"value": incoming})
    with pytest.raises(RuntimeError, match="value writer failed"):
        store.close()
    assert DeltaTable(store.values()).version() == version
    assert store.query().select() == [{"value": initial}]


def test_sparse_columns_nulls_and_initial_common_types(tmp_path):
    store = ObjectStore.local(tmp_path)
    node = Node(name="sparse", nonce="abcdef")
    store.value(node, {"number": 1, "label": 2, "nullable": None})
    store.value(node.next(), {"number": 1.5, "label": "text", "nullable": 3})
    store.close()
    store.start()
    store.value(node.next().next(), {"new": True, "nullable": None})
    store.close()
    rows = sorted(
        DeltaTable(store.values()).to_pyarrow_table().to_pylist(),
        key=lambda r: r["_x_seq"],
    )
    assert [r["number"] for r in rows] == [1.0, 1.5, None]
    assert [r["label"] for r in rows] == ["2", "text", None]
    assert [r["nullable"] for r in rows] == [None, 3, None]
    assert [r["new"] for r in rows] == [None, None, True]
    assert store.query().select() == [
        {"number": 1.0, "label": "2"},
        {"number": 1.5, "label": "text", "nullable": 3},
        {"new": True},
    ]


def test_unknown_all_null_column_needs_a_type(tmp_path):
    store = ObjectStore.local(tmp_path)
    store.value(Node(name="null"), {"unknown": None})
    with pytest.raises(RuntimeError) as raised:
        store.close()
    assert "all-null column 'unknown'" in str(raised.value.__cause__)


def test_queries_use_only_the_committed_snapshot(tmp_path, monkeypatch):
    store = ObjectStore.local(tmp_path)
    node = Node(name="snapshot", nonce="abcdef")
    store.value(node, {"value": 1, "_x_finished": True})
    store.close()
    # Uncommitted Parquet must never be discovered or inferred by a reader.
    pq.write_table(pa.table({"garbage": [True]}), store.values() / "orphan.parquet")
    query = store_module._chdb_query
    calls = 0

    def append_during_query(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            store.start()
            store.value(node, {"value": 2, "new": "later"})
            store.close()
        return query(*args, **kwargs)

    monkeypatch.setattr(store_module, "_chdb_query", append_during_query)
    assert store.query().finished().select() == [{"value": 1}]
    assert calls == 2
    assert store.query().select() == [{"value": 2, "new": "later"}]


def test_transient_snapshot_read_is_retried(tmp_path, monkeypatch):
    store = ObjectStore.local(tmp_path)
    store.value(Node(name="read-retry"), {"value": 1})
    store.close()
    table = DeltaTable(store.values())
    opening = Mock(side_effect=[OSError(errno.ETIMEDOUT, "timed out"), table])
    monkeypatch.setattr(store_module, "DeltaTable", opening)
    monkeypatch.setattr(store_module, "_DELTA_IO_RETRY_SECONDS", 0)

    assert store.query().select() == [{"value": 1}]
    assert opening.call_count == 2


def test_transient_delta_write_is_retried(tmp_path, monkeypatch):
    store = ObjectStore.local(tmp_path)
    store.value(Node(name="seed"), {"value": 0})
    store.close()
    write = store_module.write_deltalake
    attempts = 0

    def flaky_write(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EIO, "transient input/output error")
        return write(*args, **kwargs)

    monkeypatch.setattr(store_module, "write_deltalake", flaky_write)
    monkeypatch.setattr(store_module, "_DELTA_IO_RETRY_SECONDS", 0)
    store.start()
    store.value(Node(name="write-retry"), {"value": 1})
    store.close()

    assert attempts == 2
    assert store.query().name("write-retry").select() == [{"value": 1}]


def test_same_schema_commit_conflict_is_retried(tmp_path, monkeypatch):
    store = ObjectStore.local(tmp_path)
    store.value(Node(name="seed"), {"value": 0})
    store.close()
    write = store_module.write_deltalake
    attempts = 0

    def concurrent_write(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise CommitFailedError("Metadata changed since last commit.")
        return write(*args, **kwargs)

    monkeypatch.setattr(store_module, "write_deltalake", concurrent_write)
    monkeypatch.setattr(store_module, "_DELTA_IO_RETRY_SECONDS", 0)
    store.start()
    store.value(Node(name="retry"), {"value": 1})
    store.close()

    assert attempts == 2
    assert store.query().name("retry").select() == [{"value": 1}]


def test_schema_commit_conflicts_propagate_without_retry(tmp_path, monkeypatch):
    conflict = CommitFailedError("concurrent metadata changed")
    pending = Mock(side_effect=conflict)
    monkeypatch.setattr(store_module, "write_deltalake", pending)
    store = ObjectStore.local(tmp_path)
    store.value(Node(name="retry"), {"value": 1})
    with pytest.raises(RuntimeError) as raised:
        store.close()
    assert raised.value.__cause__ is conflict
    assert pending.call_count == 1


@pytest.mark.parametrize("created", [False, True])
def test_competing_schemas_keep_first_commit_and_fail_loser(
    tmp_path, monkeypatch, created
):
    store = ObjectStore.local(tmp_path)
    if created:
        store.value(Node(name="seed"), {"shared": 0})
        store.close()
    stores = [ObjectStore.local(tmp_path) for _ in range(2)]
    barrier = Barrier(2)
    write = store_module.write_deltalake

    def concurrent_write(*args, **kwargs):
        if created:
            barrier.wait(timeout=10)
        return write(*args, **kwargs)

    touch = Path.touch

    def concurrent_creation(path, *args, **kwargs):
        if path.name == ".creating":
            barrier.wait(timeout=10)
        return touch(path, *args, **kwargs)

    monkeypatch.setattr(store_module, "write_deltalake", concurrent_write)
    if not created:
        monkeypatch.setattr(Path, "touch", concurrent_creation)
    for index, writer in enumerate(stores):
        writer.value(Node(name=f"writer-{index}"), {f"column-{index}": index})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(writer.close) for writer in stores]
        outcomes = [future.exception() for future in futures]
    store.close()
    assert sum(error is None for error in outcomes) == 1
    assert sum(isinstance(error, RuntimeError) for error in outcomes) == 1
    winner = outcomes.index(None)
    rows = store.query().select()
    assert {f"column-{winner}": winner} in rows
    assert len(rows) == 1 + int(created)
    assert (
        f"column-{1 - winner}"
        not in pa.schema(DeltaTable(store.values()).schema().to_arrow()).names
    )
    assert not (store.values() / ".creating").exists()


def test_process_writers_append_same_schema_without_lost_rows(tmp_path):
    store = ObjectStore.local(tmp_path)
    store.value(Node(name="seed"), {"shared": 0})
    store.close()
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    processes = [
        context.Process(target=publish_values, args=(str(tmp_path), i, barrier))
        for i in range(3)
    ]
    for process in processes:
        process.start()
    reader = ObjectReader.local(tmp_path)
    deadline = monotonic() + 90
    try:
        previous = 0
        while any(p.is_alive() for p in processes) and monotonic() < deadline:
            rows = reader.query().select()
            assert len(rows) >= previous
            previous = len(rows)
            sleep(0.02)
        for process in processes:
            process.join(timeout=1)
            assert process.exitcode == 0
        rows = [
            (n, v)
            for n, v in reader.query().select(return_nodes=True)
            if n.name != "seed"
        ]
        assert len(rows) == 72
        for node, values in rows:
            assert values == {
                "shared": node.seq % 4,
            }
        assert DeltaTable(reader.values()).to_pyarrow_table().num_rows == 73
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
