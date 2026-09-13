"""Keyboard-accessible bootstrap-log selection for one run."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import run, ui

from theseus.cli.interface.age import AgeLabel
from theseus.cli.interface.data import RunData, RunKey
from theseus.cli.interface.log_data import LogFile
from theseus.cli.interface.log_view import LogScreen
from theseus.cli.interface.status import RunStatus
from theseus.cli.interface.navigation import Button, Screen

if TYPE_CHECKING:
    from theseus.cli.interface.driver import TheseusInterface


class LogRow(Button):
    """A home-style file row with right-aligned modification age."""

    def __init__(self, screen: LogListScreen, file: LogFile) -> None:
        self.screen = screen
        self.file = file
        super().__init__("", self.open, classes="tree-row")
        with self:
            self.age = AgeLabel()
        self.set(file)

    def set(self, file: LogFile) -> None:
        self.file = file
        self.set_text(f"  {file.path.name}")
        self.age.set(file.modified)
        self.props["aria-label"] = file.path.name

    def open(self) -> None:
        self.screen.open(self.file)


class LogListScreen(Screen):
    """List the local new-execute logs matching one stored run."""

    def __init__(
        self,
        app: TheseusInterface,
        data: RunData,
        directory: Path,
        files: tuple[LogFile, ...],
    ) -> None:
        super().__init__(app)
        self.data = data
        self.directory = directory
        self.files: tuple[LogFile, ...] = ()
        self.buttons: dict[Path, LogRow] = {}
        with self:
            with ui.element("div").classes("row-line"):
                self.back = Button("‹", app.pop_screen)
                ui.label(f"{data.name}  {data.nonce}").classes("grow")
                RunStatus(directory, data)
                ui.label("logs").classes("muted")
            ui.label(str(directory)).classes("log-metadata")
            self.empty = ui.label("no matching log files").classes("muted")
            self.list = ui.element("div").classes("log-list scroll key-list")
            self.timer = ui.timer(2, self.refresh, immediate=False)
        self.apply(files)
        if not files:
            self.back.run_method("focus")

    @property
    def location(self) -> dict[str, str]:
        return {
            "screen": "logs",
            "run": self.data.name,
            "nonce": self.data.nonce,
        }

    def apply(self, files: tuple[LogFile, ...]) -> None:
        focus_first = not self.files and bool(files)
        self.files = files
        paths = {file.path for file in files}
        for path in set(self.buttons) - paths:
            self.buttons.pop(path).delete()
        with self.list:
            for index, file in enumerate(files):
                row = self.buttons.get(file.path)
                if row is None:
                    row = self.buttons[file.path] = LogRow(self, file)
                else:
                    row.set(file)
                row.move(self.list, target_index=index)
        self.empty.set_visibility(not files)
        if focus_first:
            self.buttons[files[0].path].run_method("focus")

    def open(self, file: LogFile) -> None:
        self.timer.deactivate()
        self.app.push_screen(LogScreen, self.data, file.path)

    async def refresh(self) -> None:
        try:
            files = await run.io_bound(
                LogFile.discover,
                self.directory,
                RunKey(self.data.name, self.data.nonce),
                self.data.executions,
            )
        except OSError:
            return
        if files is not None:
            self.apply(files)

    def resume(self) -> None:
        super().resume()
        self.timer.activate()
