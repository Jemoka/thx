"""Saved views for a run or marked workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from theseus.cli.interface.data import RunData
from theseus.cli.interface.state import View

if TYPE_CHECKING:
    from theseus.cli.interface.driver import TheseusInterface

from functools import partial

from nicegui.element import Element

from theseus.cli.interface.navigation import Button


class ViewList(Element):
    def __init__(self, app: TheseusInterface, data: RunData | None = None) -> None:
        super().__init__("div")
        self.app, self.data = app, data
        self.classes("view-list scroll key-list")
        self.refresh()

    def refresh(self) -> None:
        self.clear()
        scope = (self.data.name, self.data.nonce) if self.data is not None else ()
        with self:
            for view in self.app.state.views(self.app.root_path, *scope):
                Button(view.name, partial(self.open, view)).classes("list-row")
            Button("+", self.open).classes("list-row")

    async def open(self, view: View | None = None) -> None:
        from theseus.cli.interface.view import ViewScreen

        scope = (self.data.name, self.data.nonce) if self.data is not None else ()
        if view is None:
            view = self.app.state.add_view(self.app.root_path, *scope)
        data = self.data if self.data is not None else await self.app.workspace()
        if data is not None:
            self.app.push_screen(ViewScreen, data, view)
