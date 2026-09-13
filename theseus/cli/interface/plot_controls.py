"""Terminal axis popovers, whole-label toggles, and smoothing track."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from nicegui import ui
from nicegui.element import Element
from nicegui.events import GenericEventArguments

from theseus.cli.interface.navigation import Button


class AxisPicker(Element):
    def __init__(
        self,
        axis: str,
        keys: tuple[str, ...],
        value: str | None,
        changed: Callable[[], None],
    ) -> None:
        super().__init__("div")
        self.axis, self.value, self.changed = axis, value, changed
        self.classes("axis-picker")
        with self:
            self.button = Button(f"{axis}: {value or '—'}", self.open)
            self.options = ui.element("div").classes("axis-options key-list")
            with self.options:
                for key in keys:
                    Button(key, partial(self.select, key)).classes("list-row")
            self.options.set_visibility(False)

    def open(self) -> None:
        self.options.set_visibility(not self.options.visible)
        if self.options.visible:
            self.options.default_slot.children[0].run_method("focus")

    def close(self) -> None:
        self.options.set_visibility(False)

    def select(self, value: str | None) -> None:
        self.value = value
        self.button.set_text(f"{self.axis}: {value or '—'}")
        self.close()
        self.button.run_method("focus")
        self.changed()


class PlotToggle(Button):
    def __init__(self, label: str, value: bool, changed: Callable[[], None]) -> None:
        self.toggle_label, self.value, self.changed = label, value, changed
        super().__init__(
            f"{'●' if value else '○'} {label}", self.toggle, classes="plot-toggle"
        )
        self.classes(add="enabled" if value else "")

    def toggle(self) -> None:
        self.value = not self.value
        self.set_text(f"{'●' if self.value else '○'} {self.toggle_label}")
        self.classes(
            add="enabled" if self.value else "", remove="" if self.value else "enabled"
        )
        self.changed()


class SmoothingSlider(Element):
    def __init__(self, value: float, changed: Callable[[], None]) -> None:
        super().__init__("div")
        self.value, self.changed = value, changed
        self.classes("smoothing row-line")
        with self:
            self.label = ui.label(f"smooth {value:.2f}")
            self.track = ui.element("input")
            self.track.props.update(
                {
                    "type": "range",
                    "min": 0,
                    "max": 1,
                    "step": "any",
                    "value": value,
                    "aria-label": "smoothing",
                }
            )
            self.track.on(
                "input",
                self.select,
                js_handler="e=>emit(Number(e.target.value))",
                throttle=0.05,
            )
            self.track.on(
                "keydown",
                self.select,
                js_handler="""e => {
                    const values={ArrowLeft:Number(e.target.value)-.05,ArrowRight:Number(e.target.value)+.05,Home:0,End:1};
                    if(e.key in values){e.preventDefault();e.target.value=Math.max(0,Math.min(1,values[e.key]));emit(Number(e.target.value));}
                }""",
            )

    def select(self, event: GenericEventArguments) -> None:
        self.value = max(0, min(1, event.args))
        self.track.props["value"] = self.value
        self.label.set_text(f"smooth {self.value:.2f}")
        self.changed()
