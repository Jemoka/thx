"""Per-client NiceGUI navigation and serialized background data refresh."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Concatenate, ParamSpec, TypeVar

from nicegui import run, ui
from nicegui.events import GenericEventArguments

from theseus.cli.interface.cache import ObjectCache
from theseus.cli.interface.data import RunData, RunKey, RunSnapshot, WorkspaceData
from theseus.cli.interface.location import Location
from theseus.cli.interface.log_data import LogFile
from theseus.cli.interface.log_list import LogListScreen
from theseus.cli.interface.navigation import Screen
from theseus.cli.interface.plot import PlotScreen
from theseus.cli.interface.run import RunScreen
from theseus.cli.interface.state import InterfaceState
from theseus.cli.interface.tree import RunTree
from theseus.cli.interface.view import ViewScreen

P = ParamSpec("P")
S = TypeVar("S", bound=Screen)


class TheseusInterface:
    def __init__(self, root: Path, logs: Path | None = None) -> None:
        self.client = ui.context.client
        self.restoring = True
        self.initial_location = Location(ui.context.client.request.url.query)
        self.root_path = root
        self.logs_path = logs
        self.state = InterfaceState()
        self.cache = ObjectCache(root)
        self.version = -1
        self.runs: dict[tuple[str, str], int] = {}
        self.loaded: dict[RunKey, RunData] = {}
        self.lock = asyncio.Lock()
        self.stack: list[Screen] = []
        ui.add_css(Path(__file__).with_name("terminal.css").read_text())
        for filename in ("chart.js", "keyboard.js"):
            ui.add_body_html(
                "<script>"
                + Path(__file__).with_name(filename).read_text()
                + "</script>"
            )
        self.container = ui.element("main").classes("terminal")
        ui.on("terminalkey", self.key)
        ui.on("locationchanged", self.restore_location)
        self.home = self.push_screen(RunTree)
        self.timer = ui.timer(2, self.refresh_data, immediate=False)
        ui.timer(0.01, self.restore_location, once=True)
        ui.context.client.on_delete(self.close)

    def push_screen(
        self,
        screen: Callable[Concatenate[TheseusInterface, P], S],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> S:
        if self.stack:
            self.stack[-1].set_visibility(False)
        with self.container:
            instance = screen(self, *args, **kwargs)
        self.stack.append(instance)
        if hasattr(self, "home"):
            self.update_location()
        return instance

    def pop_screen(self) -> None:
        if len(self.stack) > 1:
            self.stack.pop().delete()
            self.stack[-1].resume()
            self.update_location()
            self.client.run_javascript(
                'const screen=document.querySelector(".screen:not(.hidden)");(document.getElementById(screen.dataset.lastFocus)||screen.querySelector("button, input"))?.focus()'
            )

    async def open_run(self, key: RunKey, seq: int | None = None) -> RunScreen:
        async with self.lock:
            data = await self._load_run(key)
        return self.push_screen(RunScreen, data, seq)

    async def open_logs(
        self, data: RunData, file_name: str | None = None
    ) -> LogListScreen:
        if self.logs_path is None:
            raise ValueError("log browsing is not configured")
        files = await run.io_bound(
            LogFile.discover,
            self.logs_path,
            RunKey(data.name, data.nonce),
            data.executions,
        )
        if files is None:
            raise asyncio.CancelledError
        screen = self.push_screen(LogListScreen, data, self.logs_path, files)
        if file_name is not None:
            file = next(file for file in files if file.path.name == file_name)
            screen.open(file)
        return screen

    async def workspace(self) -> WorkspaceData | None:
        async with self.lock:
            for key in self.home.marked:
                await self._load_run(key)
            return (
                WorkspaceData(
                    tuple(
                        self.loaded[key]
                        for key in sorted(
                            self.home.marked, key=lambda key: (key.name, key.nonce)
                        )
                    )
                )
                if self.home.marked
                else None
            )

    async def _load_run(self, key: RunKey) -> RunData:
        if key not in self.loaded:
            data = await run.io_bound(RunData, self.cache.reader, key.name, key.nonce)
            if data is None:
                raise asyncio.CancelledError
            self.loaded[key] = data
        return self.loaded[key]

    def read_data(
        self,
    ) -> (
        tuple[int, dict[tuple[str, str], int], list[tuple[RunData, RunSnapshot]]] | None
    ):
        version = self.cache.poll()
        if version == self.version:
            return None
        runs: dict[tuple[str, str], int] = {}
        for node, values in self.cache.reader.query().select(
            return_nodes=True, raw=True, keys=["_x_write"]
        ):
            modified = values.get("_x_write")
            if isinstance(modified, (float, int, str)):
                key = (node.name, node.nonce)
                runs[key] = max(runs.get(key, 0), int(modified))
        return (
            version,
            runs,
            [(data, data.read(self.cache.reader)) for data in self.loaded.values()],
        )

    async def refresh_data(self) -> None:
        try:
            async with self.lock:
                result = await run.io_bound(self.read_data)
                if result is None:
                    return
                self.version, self.runs, snapshots = result
                for data, snapshot in snapshots:
                    data.apply(snapshot)
                for screen in self.stack:
                    screen.refresh_data()
        except Exception as error:
            ui.notify(f"Unable to refresh runs: {error}", type="negative")

    def update_location(self, *, replace: bool = False) -> None:
        if self.restoring:
            return
        url = Location().url(self.stack[-1], self.home)
        method = "replaceState" if replace else "pushState"
        self.client.run_javascript(
            f'if(window.location.pathname+window.location.search !== {json.dumps(url)}) history.{method}(null,"",{json.dumps(url)})'
        )

    async def restore_location(
        self, event: GenericEventArguments | None = None
    ) -> None:
        self.restoring = True
        location = Location(event.args) if event is not None else self.initial_location
        try:
            await self.refresh_data()
            while len(self.stack) > 1:
                self.stack.pop().delete()
            self.home.resume()
            self.home.marked = {
                key for key in location.marked if (key.name, key.nonce) in self.runs
            }
            self.home.expanded = location.expanded
            self.home.search = location.values.get("search", "")
            self.home.search_input.props["value"] = self.home.search
            self.home.refresh_data()
            self.home.refresh_marks()
            values = location.values
            data: RunData | WorkspaceData | None = None
            if values.get("run") and values.get("nonce"):
                key = RunKey(values["run"], values["nonce"])
                if (key.name, key.nonce) not in self.runs:
                    raise ValueError("run is no longer available")
                screen = await self.open_run(key, int(values.get("seq", "0")))
                data = screen.data
                if values.get("screen") in ("logs", "log"):
                    await self.open_logs(
                        data,
                        values.get("log") if values.get("screen") == "log" else None,
                    )
            elif values.get("screen") in ("view", "plot"):
                data = await self.workspace()
            if values.get("screen") in ("view", "plot") and data is not None:
                scope = (data.name, data.nonce) if isinstance(data, RunData) else ()
                view = next(
                    v
                    for v in self.state.views(self.root_path, *scope)
                    if v.id == int(values["view"])
                )
                self.push_screen(ViewScreen, data, view)
                if values.get("screen") == "plot":
                    plot = next(
                        p for p in self.state.plots(view) if p.id == int(values["plot"])
                    )
                    self.push_screen(PlotScreen, data, plot, view)
        except (ValueError, TypeError, KeyError, StopIteration) as error:
            ui.notify(f"Unable to restore location: {error}", type="negative")
        finally:
            self.restoring = False
            self.update_location(replace=True)

    def key(self, event: GenericEventArguments) -> None:
        self.stack[-1].key(event.args)

    def close(self) -> None:
        self.timer.cancel()
        self.state.close()
