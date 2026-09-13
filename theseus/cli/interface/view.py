"""Saved view screen containing vector plot previews."""

from __future__ import annotations

from typing import TYPE_CHECKING

from theseus.cli.interface.data import RunData, WorkspaceData
from theseus.cli.interface.state import View

if TYPE_CHECKING:
    from theseus.cli.interface.driver import TheseusInterface

from nicegui import ui

from theseus.cli.interface.actions import EditActions
from theseus.cli.interface.location import view_location
from theseus.cli.interface.navigation import Button, Screen
from theseus.cli.interface.plot import PlotPreview, PlotScreen
from theseus.cli.interface.walk import WalkButton


class ViewScreen(Screen):
    def __init__(
        self, app: TheseusInterface, data: RunData | WorkspaceData, view: View
    ) -> None:
        super().__init__(app)
        self.data, self.view = data, view
        with self:
            with ui.element("div").classes("row-line"):
                Button("‹", app.pop_screen)
                self.title = ui.label(view.name).classes("grow")
                if data.comparison:
                    WalkButton(data)
                EditActions(self.rename, self.delete_view)
            self.previews = ui.element("div").classes("scroll")
        self.resume()

    @property
    def location(self) -> dict[str, str]:
        return view_location(self.data, self.view)

    def resume(self) -> None:
        super().resume()
        self.previews.clear()
        with self.previews:
            for plot in self.app.state.plots(self.view):
                PlotPreview(self.app, self.data, plot, self.view)
            Button("+", self.add_plot)

    def add_plot(self) -> None:
        self.app.push_screen(
            PlotScreen, self.data, self.app.state.add_plot(self.view), self.view
        )

    def rename(self, name: str) -> None:
        self.view = self.app.state.rename_view(self.view, name)
        self.title.set_text(name)

    def delete_view(self) -> None:
        self.app.state.delete_view(self.view)
        self.app.pop_screen()

    def refresh_data(self) -> None:
        for preview in self.previews.default_slot.children:
            if isinstance(preview, PlotPreview):
                preview.chart.refresh()
