"""Copyable node identity and independently expandable step fields."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from pathlib import Path

from nicegui import ui
from nicegui.element import Element

from theseus.cli.interface.navigation import Button
from theseus.cli.interface.plot_math import format_timestamp


class NodeIdentity(Button):
    def __init__(self) -> None:
        super().__init__("", self.copy, classes="node-identity")
        self.node_id = ""

    def set(self, node_id: str) -> None:
        self.node_id = node_id
        self.set_text(node_id)

    def copy(self) -> None:
        ui.clipboard.write(self.node_id)
        self.classes(add="copied")
        ui.timer(0.25, lambda: self.classes(remove="copied"), once=True)


class StepFields(Element):
    def __init__(self) -> None:
        super().__init__("div")
        self.classes("fields scroll")

    def set(self, fields: Mapping[str, object]) -> None:
        self.clear()
        with self:
            for key, value in sorted(fields.items()):
                if key == "timestamp" and isinstance(value, Real):
                    value = format_timestamp(float(value))
                elif isinstance(value, Path):
                    value = str(value)
                with ui.element("details"):
                    with ui.element("summary"):
                        ui.label(key)
                    ui.label(repr(value)).classes("field-value")
