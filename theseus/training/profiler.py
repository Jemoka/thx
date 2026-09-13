"""Worker HTTP capture requests, serviced one optimizer step at a time."""

from collections.abc import Iterator
from concurrent.futures import Future, TimeoutError
from contextlib import contextmanager
from functools import partial
import gzip
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory, mkdtemp
from threading import Lock, Thread
from zipfile import ZIP_DEFLATED, ZipFile

import bottle
from cheroot.wsgi import Server
import jax
import jaxlib._profiler as _profiler
from loguru import logger

from theseus.base import Node


class Profiler:
    """Own one-step capture requests and their downloadable traces."""

    def __init__(self, port: int | None = None, *, node: Node | None = None) -> None:
        if port is None:
            port = int(os.environ.get("THESEUS_PROFILER_PORT", "0"))
        self.node = node
        self.host = socket.gethostname()
        self.process = jax.process_index()
        self.port = port
        self._files = TemporaryDirectory(prefix="theseus-profile-")
        self._capture = Lock()
        self._pending: Future[tuple[_profiler.ProfilerSession, bytes, str]] | None = (
            None
        )
        self._thread: Thread | None = None
        self._closed = False
        self._app = bottle.Bottle()
        self._app.get("/", callback=self.index)
        self._app.post("/profile", callback=self.profile)
        self._app.get("/trace/<capture>", callback=self.trace)
        self._app.get("/xprof/<capture>", callback=partial(self.trace, native=True))
        self._server = Server(
            ("0.0.0.0", port), self._app, numthreads=4, shutdown_timeout=95
        )

    def index(self) -> str:
        return str(
            bottle.template(
                Path(__file__).with_suffix(".html").read_text(),
                host=self.host,
                process=self.process,
                node=self.node,
            )
        )

    def profile(self) -> dict[str, str]:
        with self._capture:
            if self._closed:
                bottle.response.status = 503
                return {"error": "The profiler is closing."}
            if jax.process_count() != 1:
                bottle.response.status = 503
                return {
                    "error": "One-step capture currently requires a single JAX process."
                }
            if self._pending is not None and not self._pending.done():
                bottle.response.status = 409
                return {"error": "A capture is already running on this worker."}
            pending = self._pending = Future()
        try:
            session, data, label = pending.result(timeout=90)
            folder = Path(mkdtemp(prefix="capture-", dir=self._files.name))
            # Collection is already stopped. Export stays on this HTTP worker.
            session.export(data, str(folder))
            (source,) = folder.glob("plugins/profile/*/*.trace.json.gz")
            with gzip.open(source, "rt") as stream:
                trace = json.load(stream)
            trace.pop("metadata", None)
            with gzip.open(folder / "trace.json.gz", "wt") as stream:
                json.dump(trace, stream)
            if not list(folder.glob("plugins/profile/*/*.xplane.pb")):
                raise FileNotFoundError("Capture produced no native XProf profile.")
            with ZipFile(folder / "profile.zip", "w", ZIP_DEFLATED) as archive:
                for file in (folder / "plugins").rglob("*"):
                    if file.is_file():
                        archive.write(file, file.relative_to(folder))
            return {
                "trace": f"/trace/{folder.name}",
                "xprof": f"/xprof/{folder.name}",
                "label": label,
            }
        except TimeoutError:
            pending.cancel()  # Only an unclaimed request can be cancelled.
            bottle.response.status = 504
            return {"error": "Timed out waiting for a training step to finish."}
        except Exception as error:
            bottle.response.status = 502
            return {"error": str(error)}

    @contextmanager
    def measure_step(self) -> Iterator[None]:
        """Called by the training thread; synchronize only a requested step."""
        pending = self._pending
        if pending is None or pending.done() or pending.running():
            yield
            return
        with self._capture:
            claimed = pending.set_running_or_notify_cancel()
        if not claimed:
            yield
            return
        # These synchronization APIs lack annotations in the pinned JAX version.
        try:
            jax.block_until_ready(jax.live_arrays())  # type: ignore[no-untyped-call]
            jax.effects_barrier()  # type: ignore[no-untyped-call]
            label = self.node.serialize() if self.node is not None else self.host
            # The pinned JAX API separates collection from the expensive export.
            session = _profiler.ProfilerSession()
        except Exception as error:
            pending.set_exception(error)
            yield
            return
        try:
            yield
        except BaseException as error:
            pending.set_exception(error)
            raise
        finally:
            try:
                jax.block_until_ready(jax.live_arrays())  # type: ignore[no-untyped-call]
                jax.effects_barrier()  # type: ignore[no-untyped-call]
                data = session.stop()
            except Exception as error:
                if not pending.done():
                    pending.set_exception(error)
                raise
            if not pending.done():
                pending.set_result((session, data, label))

    def trace(self, capture: str, *, native: bool = False) -> bottle.HTTPResponse:
        folder = Path(self._files.name) / capture
        if folder.parent != Path(self._files.name) or not capture.startswith(
            "capture-"
        ):
            return bottle.HTTPError(404, "Unknown capture.")
        if native:
            return bottle.static_file(
                "profile.zip",
                root=str(folder),
                mimetype="application/zip",
                download=f"{capture}.zip",
            )
        trace = folder / "trace.json.gz"
        if not trace.is_file():
            return bottle.HTTPError(404, "Unknown capture.")
        return bottle.static_file(
            trace.name,
            root=str(trace.parent),
            mimetype="application/json",
            download=f"{capture}.json",
            headers={"Content-Encoding": "gzip"},
        )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("The profiler is closed.")
        if self._thread is not None:
            return
        self._server.prepare()
        self.port = int(self._server.socket.getsockname()[1])
        self._thread = Thread(target=self._server.serve, name="profiler", daemon=True)
        self._thread.start()
        logger.info(
            "PROFILE | http://{}:{} | process={}",
            self.host,
            self.port,
            self.process,
        )

    def close(self) -> None:
        if self._closed:
            return
        with self._capture:
            self._closed = True
            if (
                self._pending is not None
                and not self._pending.done()
                and not self._pending.running()
            ):
                self._pending.set_exception(
                    RuntimeError("Training closed before capture.")
                )
        try:
            if self._thread is not None:
                self._server.stop()
                self._thread.join()
        finally:
            self._files.cleanup()
