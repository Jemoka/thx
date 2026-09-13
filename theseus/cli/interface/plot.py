"""Saved plot previews and configuration screen."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nicegui.events import GenericEventArguments

from theseus.cli.interface.data import RunData, WorkspaceData
from theseus.cli.interface.navigation import KeyEvent
from theseus.cli.interface.state import Plot, View

if TYPE_CHECKING:
    from theseus.cli.interface.driver import TheseusInterface

from nicegui import ui
from nicegui.element import Element

from theseus.cli.interface.actions import EditActions
from theseus.cli.interface.chart import PlotCanvas
from theseus.cli.interface.location import view_location
from theseus.cli.interface.navigation import Button, Screen
from theseus.cli.interface.plot_controls import AxisPicker, PlotToggle, SmoothingSlider


class PlotPreview(Element):
    def __init__(
        self,
        app: TheseusInterface,
        data: RunData | WorkspaceData,
        plot: Plot,
        view: View,
    ) -> None:
        super().__init__("div")
        self.classes("plot-preview").props("tabindex=0 role=button")
        self.on("click", lambda: app.push_screen(PlotScreen, data, plot, view))
        self.on("keydown.enter", lambda: app.push_screen(PlotScreen, data, plot, view))
        with self:
            ui.label(plot.name).classes("preview-title")
            self.chart = PlotCanvas(app, data, plot, preview=True)


class PlotScreen(Screen):
    def __init__(
        self,
        app: TheseusInterface,
        data: RunData | WorkspaceData,
        plot: Plot,
        view: View,
    ) -> None:
        super().__init__(app)
        self.data, self.plot, self.view = data, plot, view
        keys = tuple(dict.fromkeys(("step", *data.scalar_keys)))
        with self:
            with ui.element("div").classes("row-line"):
                Button("‹", app.pop_screen)
                self.title = ui.label(plot.name).classes("grow bold")
                EditActions(self.rename, self.delete_plot)
            with ui.element("div").classes("row-line"):
                self.x = AxisPicker("x", keys, plot.x, self.save)
                self.y = AxisPicker("y", keys, plot.y, self.save)
            with ui.element("div").classes("row-line"):
                self.log_x = PlotToggle("log x", plot.log_x, self.save)
                self.log_y = PlotToggle("log y", plot.log_y, self.save)
                self.checkpoints = PlotToggle(
                    "checkpoints", plot.checkpoints, self.save
                )
                self.smoothing = SmoothingSlider(plot.smoothing, self.save)
            self.chart = PlotCanvas(app, data, plot)
        self.on(
            "click",
            self.dismiss,
            js_handler='e=>emit(e.target.closest(".axis-picker")?.id || "")',
        )

    @property
    def location(self) -> dict[str, str]:
        return {
            **view_location(self.data, self.view),
            "screen": "plot",
            "plot": str(self.plot.id),
        }

    def dismiss(self, event: GenericEventArguments) -> None:
        for picker in (self.x, self.y):
            if event.args != f"c{picker.id}":
                picker.close()

    def key(self, event: KeyEvent) -> None:
        if event["key"] == "Escape":
            opened = [p for p in (self.x, self.y) if p.options.visible]
            if opened:
                for picker in opened:
                    picker.close()
                opened[0].button.run_method("focus")
                return
        if event["key"] == "q" and any(p.options.visible for p in (self.x, self.y)):
            return
        super().key(event)

    def save(self) -> None:
        self.plot = self.app.state.update_plot(
            self.plot,
            x=self.x.value,
            y=self.y.value,
            log_x=self.log_x.value,
            log_y=self.log_y.value,
            checkpoints=self.checkpoints.value,
            smoothing=self.smoothing.value,
        )
        self.chart.configure(self.plot)

    def rename(self, name: str) -> None:
        self.plot = self.app.state.rename_plot(self.plot, name)
        self.title.set_text(name)

    def delete_plot(self) -> None:
        self.app.state.delete_plot(self.plot)
        self.app.pop_screen()

    def refresh_data(self) -> None:
        self.chart.refresh()
