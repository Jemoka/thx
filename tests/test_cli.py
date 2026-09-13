from unittest.mock import Mock
import pytest
from typer.testing import CliRunner
from theseus.base import ExecutionSpec, Node
from theseus.cli.app import app as clix_app
import theseus.cli.ui as ui_module
from theseus.cli.interface.cache import ObjectCache
from theseus.store import ObjectStore


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


def test_object_cache_mirrors_only_changed_value_parts(tmp_path) -> None:
    first = Node(name="alpha.cache.train", nonce="aaaaaa")
    writer = ObjectStore(ExecutionSpec.local(str(tmp_path), name="ui").hardware)
    with writer.blob(first, {"loss": 1.0}) as blob:
        (blob / "state").write_text("checkpoint")
    writer.close()

    cache = ObjectCache(tmp_path)
    assert cache.poll() == 1
    assert cache.reader.query().node(first).select() == [{"blob": blob, "loss": 1.0}]
    cached_parts = set(cache.path.glob("*.parquet"))
    assert cache.poll() == 1
    assert set(cache.path.glob("*.parquet")) == cached_parts

    second = Node(name="alpha.cache.train", nonce="bbbbbb")
    writer = ObjectStore(ExecutionSpec.local(str(tmp_path), name="ui").hardware)
    writer.value(second, {"loss": 2.0})
    writer.close()
    assert cache.poll() == 2
    assert cache.reader.query().node(second).select() == [{"loss": 2.0}]

    source_parts = set((tmp_path / "objects" / "values").glob("*.parquet"))
    retired = next(
        part for part in source_parts if part.name in {p.name for p in cached_parts}
    )
    retired.unlink()
    assert cache.poll() == 3
    assert cache.reader.query().node(first).select() == []
    assert cache.reader.query().node(second).select() == [{"loss": 2.0}]


def test_object_cache_observes_parts_copied_by_another_browser(tmp_path):
    first = ObjectCache(tmp_path)
    second = ObjectCache(tmp_path)
    writer = ObjectStore(ExecutionSpec.local(str(tmp_path), name="ui").hardware)
    node = Node(name="alpha.cache.train", nonce="abcdef")
    writer.value(node, {"loss": 2.0})
    writer.close()
    assert first.poll() == 1
    assert second.poll() == 1
    assert second.reader.query().node(node).select() == [{"loss": 2.0}]
    assert second.poll() == 1
