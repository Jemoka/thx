"""Plotly line plots with terminal styling and Theseus inspection semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from nicegui.events import GenericEventArguments

from theseus.cli.interface.data import PlotData, PlotSeries
from theseus.cli.interface.state import Plot

if TYPE_CHECKING:
    from theseus.cli.interface.driver import TheseusInterface

from functools import partial
from math import isfinite

from nicegui import ui
from nicegui.element import Element

from theseus.cli.interface.data import RunKey
from theseus.cli.interface.navigation import Button
from theseus.cli.interface.plot_math import (
    format_field,
    interpolate,
    transform_points,
    twema,
)
from theseus.cli.interface.plotly_options import axis_options, trace_options

SERIES_COLORS = ("#258fa6", "#805fa3", "#c17a38", "#538862", "#ad5b72", "#536fa5")


class RenderedSeries(NamedTuple):
    source: PlotSeries
    raw: list[tuple[float, float]]
    smooth: list[tuple[float, float]]
    checkpoints: list[tuple[float, float]]
    color: str


class PlotCanvas(Element):
    def __init__(
        self, app: TheseusInterface, data: PlotData, plot: Plot, preview: bool = False
    ) -> None:
        super().__init__("div")
        self.app, self.data, self.plot, self.preview = app, data, plot, preview
        self.zoom: tuple[float, float, float, float] | None = None
        self.domain: tuple[float, float, float, float] | None = None
        self.series: list[RenderedSeries] = []
        self.classes("plot-canvas" + (" preview-chart" if preview else ""))
        with self:
            self.heading = ui.element("div").classes("chart-heading row-line bold")
            self.heading.set_visibility(not preview)
            self.graph = ui.element("div").classes("graph")
            with self.graph:
                self.chart = ui.plotly(
                    {
                        "data": [],
                        "layout": {},
                        "config": {"displayModeBar": False, "responsive": False},
                    }
                ).classes("plotly-chart")
            self.footer = ui.element("div").classes("plot-footer row-line")
        if not preview:
            self.graph.on(
                "chartgesture",
                self.gesture,
                js_handler="e => emit(e.detail)",
                throttle=0.03,
            )
        self.chart.on(
            "plotly_afterplot",
            js_handler=f'() => window.theseusChart.resize(document.getElementById("c{self.chart.id}"))'
            if preview
            else f'() => {{window.theseusChart.resize(document.getElementById("c{self.chart.id}")); window.theseusChart.mount(document.getElementById("c{self.graph.id}"));}}',
        )
        self.configure(plot)

    def configure(self, plot: Plot) -> None:
        self.plot = plot
        self.zoom = None
        self.footer.clear()
        self.refresh()

    def refresh(self) -> None:
        self.series = []
        traces = []
        plot = self.plot
        if plot.x is not None and plot.y is not None:
            for index, source in enumerate(self.data.plot_series(plot.x, plot.y, 8000)):
                points = tuple(
                    sorted(
                        (x, y) for x, y in source.points if isfinite(x) and isfinite(y)
                    )
                )
                raw = transform_points(points, plot.log_x, plot.log_y)
                smooth = transform_points(
                    twema(points, plot.smoothing), plot.log_x, plot.log_y
                )
                checkpoints = transform_points(
                    source.checkpoints, plot.log_x, plot.log_y
                )
                if not smooth:
                    continue
                color = SERIES_COLORS[index % len(SERIES_COLORS)]
                self.series.append(
                    RenderedSeries(source, raw, smooth, checkpoints, color)
                )
                for values, opacity in (
                    ((raw, 0.25), (smooth, 1)) if plot.smoothing else ((smooth, 1),)
                ):
                    traces.append(
                        trace_options(
                            values,
                            source.name,
                            color,
                            plot.log_x,
                            plot.log_y,
                            opacity=opacity,
                        )
                    )
                if plot.checkpoints and checkpoints:
                    trace = trace_options(
                        checkpoints,
                        "checkpoint",
                        color if self.data.comparison else "#37a5be",
                        plot.log_x,
                        plot.log_y,
                        markers=True,
                    )
                    trace["customdata"] = [
                        [source.run.name, source.run.nonce, seq]
                        for point, seq in zip(
                            source.checkpoints, source.checkpoint_sequences, strict=True
                        )
                        if transform_points((point,), plot.log_x, plot.log_y)
                    ]
                    traces.append(trace)
        if self.series:
            xs, ys = zip(
                *(
                    point
                    for _, raw, smooth, _, _ in self.series
                    for point in (*raw, *smooth)
                )
            )
            self.domain = self.zoom or (min(xs), max(xs), min(ys), max(ys))
        else:
            self.domain = None
        layout: dict[str, object] = {
            "paper_bgcolor": "#f7f7f7",
            "plot_bgcolor": "#f7f7f7",
            "font": {
                "family": "Menlo, Consolas, DejaVu Sans Mono, monospace",
                "size": 14,
                "color": "#2f3338",
            },
            "margin": {
                "l": 0 if self.preview else 85,
                "r": 12,
                "t": 0 if self.preview else 8,
                "b": 0 if self.preview else 30,
            },
            "showlegend": False,
            "hovermode": False,
            "dragmode": False,
        }
        layout["xaxis"] = axis_options(
            plot.x, plot.log_x, self.domain[:2] if self.domain else None, self.preview
        )
        layout["yaxis"] = axis_options(
            plot.y, plot.log_y, self.domain[2:] if self.domain else None, self.preview
        )
        self.chart.update_figure(
            {
                "data": traces,
                "layout": layout,
                "config": {
                    "displayModeBar": False,
                    "scrollZoom": False,
                    "doubleClick": False,
                    "responsive": False,
                },
            }
        )
        self.heading.clear()
        if (
            not self.preview
            and self.series
            and plot.x is not None
            and plot.y is not None
        ):
            with self.heading:
                ui.label(plot.y).classes("grow")
                for line in self.series:
                    source, color = line.source, line.color
                    label = (
                        source.name
                        if self.data.comparison
                        else f"TWEMA({plot.y})"
                        if plot.smoothing
                        else plot.y
                    )
                    ui.label(f"─ {label}").style(f"color:{color}")
                if not self.data.comparison and plot.smoothing:
                    ui.label(f"─ {plot.y}").style("color:#b9dce3")
            if not self.footer.default_slot.children:
                with self.footer:
                    ui.label(plot.x).classes("axis-name")

    async def gesture(self, event: GenericEventArguments) -> None:
        if self.domain is None:
            return
        args = event.args
        if args["kind"] == "open":
            name, nonce, seq = args["target"]
            await self.app.open_run(RunKey(name, nonce), seq)
        elif args["kind"] == "zoom":
            x_min, x_max, y_min, y_max = args["box"]
            self.zoom = (float(x_min), float(x_max), float(y_min), float(y_max))
            self.footer.clear()
            self.refresh()
        elif args["kind"] == "reset":
            self.zoom = None
            self.footer.clear()
            self.refresh()
        elif args["kind"] == "inspect":
            self.inspect(float(args["x"]))
        elif args["kind"] == "leave":
            self.footer.clear()
            self.refresh()

    def inspect(self, x: float) -> None:
        if self.plot.x is None or self.plot.y is None:
            return
        self.footer.clear()
        with self.footer:
            ui.label(
                f"{self.plot.x}={format_field(self.plot.x, 10**x if self.plot.log_x else x)}"
            )
            for source, raw, smooth, checkpoints, color in self.series:
                value = interpolate(raw, x)
                smoothed = interpolate(smooth, x)
                if value is None:
                    continue
                sequences = transform_points(source.sequences, self.plot.log_x, False)
                seq = int(min(sequences, key=lambda p: abs(p[0] - x))[1])
                label = f"{source.name + ' ' if self.data.comparison else ''}{self.plot.y}={format_field(self.plot.y, 10**value if self.plot.log_y else value)}"
                Button(label, partial(self.app.open_run, source.run, seq)).style(
                    f"color:{color if self.data.comparison else '#2f3338'};font-weight:bold"
                )
                if self.plot.smoothing and smoothed is not None:
                    ui.label(
                        f"TWEMA({self.plot.y})={format_field(self.plot.y, 10**smoothed if self.plot.log_y else smoothed)}"
                    )
                if any(abs(cx - x) < 1e-10 for cx, _ in checkpoints):
                    ui.label("checkpoint").style(f"color:{color}")
