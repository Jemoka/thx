"""Bounded, keyboard and pointer operated run timeline."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable

from nicegui import ui
from nicegui.element import Element
from nicegui.events import GenericEventArguments

from theseus.cli.interface.data import RunData


class RunTimeline(Element):
    def __init__(
        self, data: RunData, on_select: Callable[[int], None], seq: int | None = None
    ) -> None:
        super().__init__("div")
        self.data, self.on_select = data, on_select
        self.selected = data.nearest(seq if seq is not None else 0)
        self.classes("timeline").props(
            'tabindex=0 role=slider aria-label="run timeline"'
        )
        self.on(
            "click",
            self.click,
            js_handler="e => emit((e.clientX-e.currentTarget.getBoundingClientRect().left)/Math.max(1,e.currentTarget.clientWidth-1))",
        )
        self.refresh()

    def refresh(self) -> None:
        self.selected = self.data.nearest(self.selected)
        self.clear()
        timeline = self.data.timeline(512)
        self.props.update(
            {
                "aria-valuemin": timeline.first,
                "aria-valuemax": timeline.last,
                "aria-valuenow": self.selected,
            }
        )
        with self:
            for state in timeline.bins:
                ui.element("span").style(
                    "flex:1;background:"
                    + (
                        "#37a5be"
                        if state & 2
                        else "#929292"
                        if state & 1
                        else "#ebebeb"
                    )
                )
            position = (self.selected - timeline.first) / max(
                1, timeline.last - timeline.first
            )
            ui.label("│").classes("timeline-cursor").style(
                f"left:calc({position * 100}% - {position}ch)"
            )

    def select(self, seq: int) -> None:
        self.selected = self.data.nearest(seq)
        self.refresh()
        self.on_select(self.selected)

    def click(self, event: GenericEventArguments) -> None:
        if self.data.steps:
            self.select(
                round(
                    self.data.steps[0]
                    + event.args * (self.data.steps[-1] - self.data.steps[0])
                )
            )
        self.run_method("focus")

    def scrub(self, offset: int, checkpoint: bool = False) -> None:
        if not self.data.steps:
            return
        if checkpoint:
            steps = sorted(self.data.checkpoints)
            index = (
                bisect_left(steps, self.selected) - 1
                if offset < 0
                else bisect_right(steps, self.selected)
            )
            if 0 <= index < len(steps):
                self.select(steps[index])
        else:
            index = bisect_left(self.data.steps, self.selected)
            self.select(
                self.data.steps[min(len(self.data.steps) - 1, max(0, index + offset))]
            )
