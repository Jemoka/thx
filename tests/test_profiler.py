"""One-step capture, HTTP delivery, and training/export thread ownership."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO
from zipfile import ZipFile
import gzip
import json
from pathlib import Path
from threading import Event, Thread, get_ident
import time
import urllib.error
import urllib.parse
import urllib.request
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import pytest

from theseus.base import Node
from theseus.training.profiler import Profiler


@pytest.fixture
def profiler(monkeypatch):
    listener = Mock(side_effect=AssertionError("No native listener should start"))
    monkeypatch.setattr(jax.profiler, "start_server", listener)
    monkeypatch.setattr(jax.profiler, "stop_server", listener)
    server = Profiler(0, node=Node(name="<training>", seq=7))
    server.start()
    yield server
    server.close()
    listener.assert_not_called()


def request(profiler, path="/", **form):
    data = urllib.parse.urlencode(form).encode() if path == "/profile" else None
    try:
        response = urllib.request.urlopen(
            f"http://127.0.0.1:{profiler.port}{path}", data, timeout=100
        )
    except urllib.error.HTTPError as error:
        response = error
    with response:
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return response.status, body


@contextmanager
def queued_capture(profiler):
    previous = profiler._pending
    with ThreadPoolExecutor(max_workers=1) as executor:
        response = executor.submit(request, profiler, "/profile")
        deadline = time.monotonic() + 5
        while profiler._pending is previous and time.monotonic() < deadline:
            time.sleep(0.001)
        assert profiler._pending is not previous, "HTTP request was not queued"
        yield response


@pytest.fixture
def collector(monkeypatch):
    def export(data, root):
        folder = Path(root) / "plugins/profile/capture"
        folder.mkdir(parents=True)
        (folder / "host.xplane.pb").write_bytes(b"native profile data")
        with gzip.open(folder / "host.trace.json.gz", "wt") as stream:
            json.dump({"traceEvents": [], "metadata": {}}, stream)

    session = Mock()
    session.stop.return_value = b"collected trace"
    session.export.side_effect = export
    monkeypatch.setattr(
        "theseus.training.profiler._profiler.ProfilerSession",
        Mock(return_value=session),
    )
    return session


def test_capture_download_and_cleanup(profiler, collector):
    status, body = request(profiler)
    assert status == 200
    assert b"&lt;training&gt;" in body
    assert b"seconds" not in body
    root = Path(profiler._files.name)
    for seq in (7, 8):
        profiler.node.seq = seq
        with queued_capture(profiler) as response:
            with profiler.measure_step():
                pass
            status, body = response.result(timeout=5)
        assert status == 200, body
        result = json.loads(body)
        assert result["label"] == profiler.node.serialize()
        status, trace = request(profiler, result["trace"])
        assert status == 200
        assert json.loads(trace) == {"traceEvents": []}
        status, zipped = request(profiler, result["xprof"])
        assert status == 200
        with ZipFile(BytesIO(zipped)) as archive:
            assert set(archive.namelist()) == {
                "plugins/profile/capture/host.xplane.pb",
                "plugins/profile/capture/host.trace.json.gz",
            }
            assert (
                archive.read("plugins/profile/capture/host.xplane.pb")
                == b"native profile data"
            )
    assert collector.stop.call_count == 2
    assert collector.export.call_count == 2
    profiler.close()
    profiler.close()
    assert not root.exists()
    assert not profiler._thread.is_alive()


def test_ordinary_steps_do_not_synchronize(profiler, monkeypatch):
    barrier = Mock(side_effect=AssertionError("Ordinary training must not wait"))
    monkeypatch.setattr(jax, "block_until_ready", barrier)
    monkeypatch.setattr(jax, "effects_barrier", barrier)
    for _ in range(3):
        with profiler.measure_step():
            pass
    barrier.assert_not_called()


def test_collection_bounds_one_step_and_export_does_not_block_training(
    profiler, collector, monkeypatch
):
    training_thread = get_ident()
    events = []
    exporting = Event()
    release_export = Event()
    original_export = collector.export.side_effect

    def barrier(*args):
        events.append(("barrier", get_ident()))

    def stop():
        events.append(("stop", get_ident()))
        return b"collected trace"

    def export(data, root):
        events.append(("export", get_ident()))
        exporting.set()
        assert release_export.wait(5)
        original_export(data, root)

    monkeypatch.setattr(jax, "block_until_ready", barrier)
    monkeypatch.setattr(jax, "effects_barrier", Mock())
    collector.stop.side_effect = stop
    collector.export.side_effect = export
    with queued_capture(profiler) as response:
        try:
            with profiler.measure_step():
                events.append(("step", get_ident()))
            assert exporting.wait(5)
            for _ in range(3):
                with profiler.measure_step():
                    events.append(("next", get_ident()))
            assert not response.done()
        finally:
            release_export.set()
        assert response.result(timeout=5)[0] == 200
    assert [name for name, _ in events[:4]] == ["barrier", "step", "barrier", "stop"]
    assert all(thread == training_thread for name, thread in events if name != "export")
    assert (
        next(thread for name, thread in events if name == "export") != training_thread
    )
    assert collector.stop.call_count == 1


def test_overlap_is_rejected(profiler, collector):
    with queued_capture(profiler) as response:
        assert request(profiler, "/profile")[0] == 409
        with profiler.measure_step():
            assert request(profiler, "/profile")[0] == 409
        assert response.result(timeout=5)[0] == 200


def test_cancelled_request_does_not_capture(profiler, collector):
    with queued_capture(profiler) as response:
        assert profiler._pending.cancel()
        with profiler.measure_step():
            pass
        assert response.result(timeout=5)[0] == 502
    collector.stop.assert_not_called()


def test_training_exception_stops_collection(profiler, collector):
    with queued_capture(profiler) as response:
        with pytest.raises(ValueError, match="training failed"):
            with profiler.measure_step():
                raise ValueError("training failed")
        assert response.result(timeout=5)[0] == 502
    collector.stop.assert_called_once()
    collector.export.assert_not_called()


def test_export_failure_is_visible(profiler, collector):
    collector.export.side_effect = RuntimeError("export failed")
    with queued_capture(profiler) as response:
        with profiler.measure_step():
            pass
        status, body = response.result(timeout=5)
    assert status == 502
    assert json.loads(body)["error"] == "export failed"
    assert request(profiler, "/trace/not-a-capture")[0] == 404


def test_close_releases_pending_request(profiler, collector):
    with queued_capture(profiler) as response:
        profiler.close()
        assert response.result(timeout=5)[0] == 502
    collector.stop.assert_not_called()


def test_multiple_processes_are_not_partially_profiled(profiler, monkeypatch):
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    assert request(profiler, "/profile")[0] == 503
    assert profiler._pending is None


def test_start_is_idempotent_and_close_without_start_is_safe(profiler):
    thread = profiler._thread
    profiler.start()
    assert profiler._thread is thread
    unused = Profiler(0)
    root = Path(unused._files.name)
    unused.close()
    assert not root.exists()
    with pytest.raises(RuntimeError, match="closed"):
        unused.start()


def test_default_port_is_ephemeral_and_environment_can_be_overridden(monkeypatch):
    default = Profiler()
    monkeypatch.setenv("THESEUS_PROFILER_PORT", "9100")
    configured = Profiler()
    explicit = Profiler(0)
    assert default.port == 0
    assert configured.port == 9100
    assert explicit.port == 0
    default.close()
    configured.close()
    explicit.close()


def test_real_captures_leave_jax_work_progressing():
    server = Profiler(0, node=Node(name="cpu-training"))
    stop = Event()
    steps = []
    errors = []
    compute = jax.jit(lambda x: jnp.sin(x @ x) * 0.01)
    initial = compute(jnp.ones((64, 64))).block_until_ready()

    def train():
        value = initial
        try:
            while not stop.is_set():
                with server.measure_step():
                    value = compute(value)
                steps.append(time.monotonic())
                time.sleep(0.001)
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=train)
    try:
        server.start()
        worker.start()
        for _ in range(2):
            before = len(steps)
            status, body = request(server, "/profile")
            assert status == 200, body.decode()
            status, trace = request(server, json.loads(body)["trace"])
            assert status == 200
            assert json.loads(trace)["traceEvents"]
            status, zipped = request(server, json.loads(body)["xprof"])
            assert status == 200
            with ZipFile(BytesIO(zipped)) as archive:
                profiles = [
                    name for name in archive.namelist() if name.endswith(".xplane.pb")
                ]
                assert profiles
                assert all(
                    name.startswith("plugins/profile/") for name in archive.namelist()
                )
                assert all(archive.read(name) for name in profiles)
            assert len(steps) > before
            after = len(steps)
            deadline = time.monotonic() + 3
            while len(steps) == after and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(steps) > after, "JAX work stopped after capture"
            assert not errors
    finally:
        stop.set()
        worker.join(timeout=5)
        server.close()
    assert not worker.is_alive()
