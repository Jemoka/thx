from unittest.mock import Mock
import pytest
from typer.testing import CliRunner
from theseus.base import Node
from theseus.cli.app import app as clix_app
import theseus.cli.ui as ui_module
from theseus.cli.interface.driver import TheseusInterface
from theseus.cli.interface.data import RunData, RunKey
from theseus.store import ObjectReader, ObjectStore
from tests.test_store_clickhouse import clickhouse_server


@pytest.mark.parametrize("serve", [False, True])
@pytest.mark.parametrize("with_logs", [False, True])
def test_ui_launches_nicegui(tmp_path, monkeypatch, serve, with_logs):
    launch = Mock()
    interface = Mock()
    monkeypatch.setattr(ui_module.nicegui, "run", launch)
    monkeypatch.setattr(ui_module, "TheseusInterface", interface)
    logs = tmp_path / "logs"
    logs.mkdir()
    result = CliRunner().invoke(
        clix_app,
        [
            "ui",
            *(["--serve"] if serve else []),
            "--bind",
            "127.0.0.1",
            "--port",
            "9000",
            str(tmp_path),
            *(["--logs", str(logs)] if with_logs else []),
        ],
    )
    assert result.exit_code == 0, result.output
    assert launch.call_args.kwargs["host"] == "127.0.0.1"
    assert launch.call_args.kwargs["port"] == 9000
    assert launch.call_args.kwargs["show"] is (not serve)
    assert launch.call_args.kwargs["reload"] is False
    launch.call_args.kwargs["root"]()
    interface.assert_called_once_with(tmp_path, logs if with_logs else None)


def test_ui_reads_committed_store_updates(tmp_path, clickhouse_server) -> None:
    writer = ObjectStore.local(tmp_path, metadata=clickhouse_server)
    interface = TheseusInterface.__new__(TheseusInterface)
    interface.reader = ObjectReader.local(tmp_path, metadata=clickhouse_server)
    interface.loaded = {}
    assert interface.read_data() == ({}, [])

    first = Node(name="alpha.store.train", nonce="aaaaaa")
    with writer.blob(first, {"loss": 1.0}) as blob:
        (blob / "state").write_text("checkpoint")
    writer.close()
    runs, snapshots = interface.read_data()
    assert set(runs) == {(first.name, first.nonce)}
    assert snapshots == []
    data = RunData(interface.reader, first.name, first.nonce)
    assert data.rows[0][1]["blob"] == blob
    interface.loaded[RunKey(first.name, first.nonce)] = data

    # A live reader sees new commits from another connection.
    writer.start()
    writer.value(first, {"loss": 2.0})
    writer.close()
    runs, snapshots = interface.read_data()
    assert set(runs) == {(first.name, first.nonce)}
    assert len(snapshots) == 1
    data.apply(snapshots[0][1])
    assert data.rows[0][1]["loss"] == 2.0
    assert data.rows[0][1]["blob"] == blob

    writer.start()
    writer._client.command(f"TRUNCATE TABLE {writer._table}")
    writer.close()
    runs, snapshots = interface.read_data()
    assert runs == {}
    data.apply(snapshots[0][1])
    assert data.rows == {}
    interface.reader.close()
