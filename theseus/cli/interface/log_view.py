"""Live, line-numbered display for one bootstrap log."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import run, ui
from nicegui.element import Element
from nicegui.elements.log import Log

from theseus.cli.interface.data import RunData
from theseus.cli.interface.log_data import LogReader, LogUpdate
from theseus.cli.interface.navigation import Button, KeyEvent, Screen
from theseus.cli.interface.plot_math import format_timestamp

if TYPE_CHECKING:
    from theseus.cli.interface.driver import TheseusInterface


_VISIBLE_LINES = 1000


class NumberedLog(Log):
    """Append line-numbered rows while retaining NiceGUI's follow-scroll behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.classes("log-view")
        self.props("role=log aria-label=log-output tabindex=0")
        self.partial: tuple[int, str] | None = None
        self.partial_row: Element | None = None

    def append(self, update: LogUpdate) -> None:
        if update.reset:
            self.clear()
            self.partial = None
            self.partial_row = None
        if not update.lines and update.partial == self.partial:
            return
        if self.partial_row is not None:
            self.remove(self.partial_row)
            self.partial_row = None
        for number, content in update.lines:
            self._append_line(number, content)
        if update.partial is not None:
            self.partial_row = self._append_line(*update.partial, partial=True)
        self.partial = update.partial
        while len(self.default_slot.children) > _VISIBLE_LINES:
            self.remove(0)

    def scroll(self, key: str) -> None:
        amount = {
            "ArrowUp": "-20",
            "ArrowDown": "20",
            "PageUp": "-target.clientHeight",
            "PageDown": "target.clientHeight",
            "Home": "-target.scrollHeight",
            "End": "target.scrollHeight",
        }[key]
        self.client.run_javascript(
            f"const target=document.getElementById('c{self.id}')"
            ".querySelector('.q-scrollarea__container');"
            f"target.scrollBy({{top:{amount}}})"
        )

    def _append_line(self, number: int, content: str, partial: bool = False) -> Element:
        with self:
            row = ui.element("div").classes(
                "log-line" + (" log-partial" if partial else "")
            )
            with row:
                ui.label(str(number)).classes("log-line-number")
                ui.label(content or " ").classes("log-line-content")
        return row


class LogScreen(Screen):
    """Continuously follow one selected bootstrap log."""

    def __init__(self, app: TheseusInterface, data: RunData, path: Path) -> None:
        super().__init__(app)
        self.data = data
        self.path = path
        self.reader = LogReader(path, _VISIBLE_LINES)
        self.reading = False
        with self:
            with ui.element("div").classes("row-line"):
                Button("‹", app.pop_screen)
                ui.label(path.name).classes("grow bold")
                self.following = ui.label("following").classes("log-following")
            self.metadata = ui.label("loading").classes("log-metadata")
            self.output = NumberedLog()
            self.timer = ui.timer(1, self.refresh)
        self.output.run_method("focus")

    @property
    def location(self) -> dict[str, str]:
        return {
            "screen": "log",
            "run": self.data.name,
            "nonce": self.data.nonce,
            "log": self.path.name,
        }

    async def refresh(self) -> None:
        if self.reading:
            return
        self.reading = True
        try:
            update = await run.io_bound(self.reader.read)
            if update is None:
                return
            self.output.append(update)
            self.metadata.set_text(
                f"modified {format_timestamp(update.modified / 1e9)}"
                f"  ·  {update.size:,} bytes"
                f"  ·  {update.total_lines:,} lines"
            )
            self.following.set_text("following")
        except FileNotFoundError:
            self.following.set_text("waiting")
        except OSError as error:
            self.following.set_text("unavailable")
            self.metadata.set_text(str(error))
        finally:
            self.reading = False

    def key(self, event: KeyEvent) -> None:
        if event["key"] in (
            "ArrowUp",
            "ArrowDown",
            "PageUp",
            "PageDown",
            "Home",
            "End",
        ):
            self.output.scroll(event["key"])
        else:
            super().key(event)
