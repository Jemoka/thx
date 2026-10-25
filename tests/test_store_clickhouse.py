"""Persistent schema, ownership, batching, and connection failure contracts."""
import subprocess
import errno
import sys
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import Mock
from clickhouse_connect.driver.exceptions import OperationalError

import pytest
import theseus.store as store_module
from theseus.base import Node
from theseus.store import ObjectReader, ObjectStore
from tests.test_store import metadata_home


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
    store.start()
    schema = store._schema()
    store.close()
    store.start()
    store.value(node, {"value": incoming})
    store.close()
    store.start()
    assert store._schema() == schema
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

    store.start()
    store.value(node, {"value": incoming})
    with pytest.raises(RuntimeError, match="value writer failed"):
        store.close()
    reader = ObjectReader.local(tmp_path)
    reader.start()
    try:
        assert reader.query().select() == [{"value": initial}]
    finally:
        reader.close()


def test_sparse_columns_nulls_and_initial_common_types(tmp_path):
    store = ObjectStore.local(tmp_path)
    node = Node(name="sparse", nonce="abcdef")
    store.value(node, {"number": 1, "label": 2, "nullable": None})
    store.value(node.next(), {"number": 1.5, "label": "text", "nullable": 3})
    store.close()
    store.start()
    store.value(node.next().next(), {"new": True, "nullable": None})
    store.close()
    store.start()
    rows = sorted(
        store._query_arrow(f"SELECT * FROM {store._table}").to_pylist(),
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



def test_batch_and_time_flush(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "_VALUE_BATCH_SIZE", 2)
    monkeypatch.setattr(store_module, "_VALUE_FLUSH_SECONDS", 0.1)
    store = ObjectStore.local(tmp_path)
    for index in range(3):
        store.value(Node(name="batch", seq=index), {"value": index})
    deadline = monotonic() + 10
    while len(store.query().all()) < 3 and monotonic() < deadline:
        store._raise_writer_error()
        sleep(0.01)
    assert len(store.query().all()) == 3
    store.close()
    assert store.thread is None
    with pytest.raises(RuntimeError, match="closed"):
        store.value(Node(name="late"), {})
    with pytest.raises(RuntimeError, match="closed"):
        store.query().all()
    store.start()
    assert len(store.query().all()) == 3


def test_embedded_lock_is_exclusive_and_released(tmp_path):
    metadata = tmp_path / "metadata.chdb"
    first = ObjectStore.local(tmp_path, metadata=str(metadata))
    with pytest.raises(RuntimeError, match="already open"):
        ObjectStore.local(tmp_path, metadata=metadata.as_uri().replace("file:", "chdb:"))
    lock = Path(str(metadata) + ".lock")
    assert lock.exists()
    # The OS lock, not existence of its file, determines ownership.
    first.close()
    second = ObjectStore.local(tmp_path, metadata=str(metadata))
    second.value(Node(name="second"), {"value": 2})
    second.close()
    assert lock.exists()


@pytest.mark.parametrize("failure", [
    OSError("disk unavailable"),
    PermissionError(errno.EACCES, "permission denied"),
    OperationalError("authentication failed"),
    TypeError("column has fixed type String"),
])
def test_writer_failure_propagates_without_retry_and_releases_lock(tmp_path, monkeypatch, failure):
    store = ObjectStore.local(tmp_path)
    insert = Mock(side_effect=failure)
    monkeypatch.setattr(store._client, "insert", insert)
    store.value(Node(name="broken"), {"value": 1})
    with pytest.raises(RuntimeError, match="value writer failed") as raised:
        store.close()
    assert raised.value.__cause__ is failure
    assert insert.call_count == 1
    assert store._client is None
    assert store._file_lock is None
    with pytest.raises(RuntimeError, match="value writer failed"):
        store.start()
    reader = ObjectReader.local(tmp_path)
    reader.start()
    assert reader.query().all() == []


@pytest.mark.parametrize("flush", ["close", "size", "timer"])
@pytest.mark.parametrize("phase", ["schema", "insert", "acknowledgement"])
def test_transient_writer_failure_retries_same_events(tmp_path, monkeypatch, flush, phase):
    monkeypatch.setattr(store_module, "_VALUE_BATCH_SIZE", 1 if flush == "size" else 4096)
    monkeypatch.setattr(store_module, "_VALUE_FLUSH_SECONDS", 0.01 if flush == "timer" else 30)
    waits = Mock()
    monkeypatch.setattr(store_module, "sleep", waits)
    store = ObjectStore.local(tmp_path)
    method = "_schema" if phase == "schema" else "insert"
    target = store if phase == "schema" else store._client
    original = getattr(target, method)
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if phase == "acknowledgement":
                original(*args, **kwargs)
                raise OperationalError("Failed to read response data from server")
            raise OSError(errno.EINTR, "interrupted system call")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, method, fail_once)
    node = Node(name="retry", nonce="abcdef")
    store.value(node, {"value": 7})
    if flush != "close":
        deadline = monotonic() + 10
        while waits.call_count == 0 and monotonic() < deadline:
            sleep(0.01)
        assert waits.call_count == 1
    store.close()
    assert waits.call_args_list == [((0.25,), {})]
    reader = ObjectReader.local(tmp_path)
    try:
        rows = reader._query_arrow(f"SELECT * FROM {reader._table}").to_pylist()
        assert len(rows) == (2 if phase == "acknowledgement" else 1)
        assert all(row == rows[0] for row in rows)
        assert reader.query().node(node).select() == [{"value": 7}]
    finally:
        reader.close()


def test_transient_writer_failure_exhausts_bounded_retries(tmp_path, monkeypatch):
    store = ObjectStore.local(tmp_path)
    failure = OperationalError("Connection reset by peer")
    insert = Mock(side_effect=failure)
    waits = Mock()
    monkeypatch.setattr(store._client, "insert", insert)
    monkeypatch.setattr(store_module, "sleep", waits)
    store.value(Node(name="exhausted"), {"value": 1})
    with pytest.raises(RuntimeError, match="value writer failed") as raised:
        store.close()
    assert raised.value.__cause__ is failure
    assert insert.call_count == 3
    assert waits.call_args_list == [((0.25,), {}), ((0.5,), {})]
    assert store._client is None
    assert store._file_lock is None


def test_connection_failure_releases_lock(tmp_path, monkeypatch):
    connect = store_module.clickhouse_connect.get_client
    failure = Mock(side_effect=OSError("connection failed"))
    monkeypatch.setattr(store_module.clickhouse_connect, "get_client", failure)
    with pytest.raises(OSError, match="connection failed"):
        ObjectStore.local(tmp_path)
    assert failure.call_count == 1
    monkeypatch.setattr(store_module.clickhouse_connect, "get_client", connect)
    store = ObjectStore.local(tmp_path)
    store.close()


def test_cluster_metadata_selects_server_without_local_fallback(tmp_path, monkeypatch):
    from theseus.base.hardware import local

    hardware = local(str(tmp_path), "-")
    hardware.hosts[0].cluster.metadata = "https://localhost:8443/theseus"
    failure = Mock(side_effect=ConnectionError("unavailable"))
    monkeypatch.setattr(store_module.clickhouse_connect, "get_client", failure)
    with pytest.raises(ConnectionError, match="unavailable"):
        ObjectStore(hardware)
    assert failure.call_args.kwargs == {
        "dsn": "https://localhost:8443/theseus", "query_retries": 2,
    }
    assert not (tmp_path / ".thx.v2.chdb").exists()


@pytest.mark.parametrize("status,failures,attempts", [(503, 2, 3), (503, 3, 3), (401, 1, 1)])
def test_remote_query_retries_are_bounded(tmp_path, clickhouse_server, monkeypatch, status, failures, attempts):
    from urllib3.response import HTTPResponse
    from clickhouse_connect.driver.exceptions import DatabaseError

    store = ObjectStore.local(tmp_path, metadata=clickhouse_server)
    request = store._client.http.request
    calls = 0

    def transient_response(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= failures:
            return HTTPResponse(body=b"injected server failure", status=status)
        return request(*args, **kwargs)

    monkeypatch.setattr(store._client.http, "request", transient_response)
    try:
        if status == 503 and failures == 2:
            assert store._query_arrow("SELECT 7 AS value").to_pylist() == [{"value": 7}]
        else:
            with pytest.raises(DatabaseError):
                store._query_arrow("SELECT 7 AS value")
        assert calls == attempts
    finally:
        store.close()


def test_default_metadata_is_outside_object_root(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    store = ObjectStore.local(root)
    assert store._metadata == str(tmp_path / ".thx.v2.chdb")
    assert (tmp_path / ".thx.v2.chdb").is_dir()
    assert not (root / "objects" / "values").exists()

def test_root_identity_is_stable_and_namespaces_are_isolated(tmp_path):
    from uuid import UUID

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    node = Node(name="same-node", nonce="abcdef")
    first = ObjectStore.local(first_root)
    first.value(node, {"value": 1})
    first.close()
    identity = (first_root / ".id").read_bytes()
    UUID(identity.decode())
    assert not list(first_root.glob(".id.*"))
    second = ObjectStore.local(second_root)
    assert second._table != first._table
    assert second.query().all() == []
    second.value(node, {"value": 2})
    second.close()
    first.start()
    assert (first_root / ".id").read_bytes() == identity
    assert first.query().select() == [{"value": 1}]


def test_root_identity_is_not_under_custom_objects_directory(tmp_path):
    from theseus.base.hardware import local

    hardware = local(str(tmp_path), "-")
    hardware.hosts[0].cluster.objects = str(tmp_path / "elsewhere" / "blobs-root")
    store = ObjectStore(hardware)
    assert (tmp_path / ".id").is_file()
    assert not (store.root() / ".id").exists()
    store.value(Node(name="custom"), {"value": 1})
    store.close()
    reader = ObjectReader(hardware.hosts[0].cluster)
    assert reader.query().select() == [{"value": 1}]


def test_reader_requires_identity_and_store_preserves_invalid_identity(tmp_path):
    with pytest.raises(FileNotFoundError):
        ObjectReader.local(tmp_path)
    identity = tmp_path / ".id"
    identity.write_text("not a UUID")
    with pytest.raises(ValueError):
        ObjectStore.local(tmp_path)
    assert identity.read_text() == "not a UUID"
    assert not (tmp_path / ".thx.v2.chdb").exists()

def test_embedded_lock_excludes_other_processes(tmp_path):
    store = ObjectStore.local(tmp_path)
    code = '''
import sys
from theseus.store import ObjectReader
try:
    reader = ObjectReader.local(sys.argv[1], metadata=sys.argv[2])
except RuntimeError as error:
    assert 'already open' in str(error)
    print('locked')
else:
    reader.close()
    print('opened')
'''
    args = [sys.executable, "-c", code, str(tmp_path), store._metadata]
    result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=True)
    assert result.stdout.strip() == "locked"
    store.close()
    result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=True)
    assert result.stdout.strip() == "opened"


@pytest.fixture
def clickhouse_server(tmp_path):
    import os
    import signal
    import shutil
    import socket
    import urllib.request

    binary = shutil.which("clickhouse")
    if binary is None:
        pytest.skip("ClickHouse server binary is required for HTTP integration tests")
    directory = tmp_path / "server"
    directory.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = directory / "config.xml"
    config.write_text(f'''<clickhouse>
<logger><level>warning</level><console>true</console></logger>
<listen_host>127.0.0.1</listen_host><http_port>{port}</http_port>
<path>{directory}/data/</path><tmp_path>{directory}/tmp/</tmp_path>
<users><default><password></password><networks><ip>127.0.0.1</ip></networks>
<profile>default</profile><quota>default</quota></default></users>
<profiles><default><max_threads>2</max_threads></default></profiles>
<quotas><default></default></quotas>
<max_server_memory_usage>2000000000</max_server_memory_usage>
</clickhouse>''')
    with (directory / "server.log").open("w+") as log:
        server = subprocess.Popen(
            [binary, "server", f"--config-file={config}"], stdout=log, stderr=log,
            start_new_session=True,
        )
        try:
            deadline = monotonic() + 15
            while monotonic() < deadline:
                if server.poll() is not None:
                    log.seek(0)
                    pytest.fail(log.read()[-6000:])
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=.3):
                        break
                except OSError:
                    sleep(.1)
            else:
                pytest.fail("ClickHouse server startup timed out")
            yield f"http://127.0.0.1:{port}/default"
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


def test_server_concurrent_creators_share_identity_and_publish(tmp_path, monkeypatch, clickhouse_server):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from theseus.base.hardware import local

    hardware = local(str(tmp_path), "-")
    hardware.hosts[0].cluster.metadata = clickhouse_server
    barrier = Barrier(4)
    link = store_module.os.link

    def publish_identity(source, target):
        barrier.wait(timeout=10)
        return link(source, target)

    monkeypatch.setattr(store_module.os, "link", publish_identity)
    with ThreadPoolExecutor(max_workers=4) as workers:
        stores = list(workers.map(lambda _: ObjectStore(hardware), range(4)))
        assert len({store._table for store in stores}) == 1
        for index, store in enumerate(stores):
            store.value(Node(name=f"writer-{index}"), {"shared": index, f"new-{index}": float(index)})
        list(workers.map(lambda store: store.close(), stores))
    reader = ObjectReader(hardware.hosts[0].cluster)
    assert reader.query().select() == [
        {"shared": index, f"new-{index}": float(index)} for index in range(4)
    ]
    assert not list(tmp_path.glob(".id.*"))


def test_server_rejects_concurrent_incompatible_column_types(tmp_path, monkeypatch, clickhouse_server):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    first = ObjectStore.local(tmp_path, metadata=clickhouse_server)
    second = ObjectStore.local(tmp_path, metadata=clickhouse_server)
    barrier = Barrier(2)
    for store in (first, second):
        command = store._client.command

        def add_column(sql, *args, _command=command, **kwargs):
            if sql.startswith("ALTER TABLE"):
                barrier.wait(timeout=10)
            return _command(sql, *args, **kwargs)

        monkeypatch.setattr(store._client, "command", add_column)
    first.value(Node(name="integer"), {"conflict": 1})
    second.value(Node(name="string"), {"conflict": "text"})
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(store.close) for store in (first, second)]
        errors = [future.exception() for future in futures]
    assert sum(error is None for error in errors) == 1
    failure = next(error for error in errors if error is not None)
    assert isinstance(failure.__cause__, TypeError)
    reader = ObjectReader.local(tmp_path, metadata=clickhouse_server)
    rows = reader.query().select()
    assert rows in [[{"conflict": 1}], [{"conflict": "text"}]]
