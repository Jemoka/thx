"""Screen stack and terminal controls shared by the browser interface."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from theseus.cli.interface.driver import TheseusInterface

from nicegui import ui
from nicegui.element import Element


class KeyEvent(TypedDict):
    key: str
    shiftKey: bool


class Button(Element):
    """A native, keyboard-accessible single-cell-height button."""

    def __init__(
        self,
        text: str,
        on_click: Callable[[], object] | None = None,
        *,
        classes: str = "",
    ) -> None:
        super().__init__("button")
        self.props("type=button").classes(classes)
        self.label = ui.label(text).move(self)
        if on_click is not None:
            self.on("click", on_click)

    def set_text(self, text: str) -> None:
        self.label.set_text(text)


class Screen(Element):
    """One retained screen in a per-browser navigation stack."""

    def __init__(self, app: TheseusInterface) -> None:
        super().__init__("section")
        self.app = app
        self.classes("screen")

    @property
    def location(self) -> dict[str, str]:
        return {}

    def resume(self) -> None:
        self.set_visibility(True)

    def refresh_data(self) -> None:
        """Update projections after the shared cache changes."""

    def key(self, event: KeyEvent) -> None:
        if event["key"] in ("Escape", "q"):
            self.app.pop_screen()
