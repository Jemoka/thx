"""Inline rename and two-press deletion controls."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui
from nicegui.element import Element
from nicegui.events import GenericEventArguments

from theseus.cli.interface.navigation import Button


class EditActions(Element):
    def __init__(
        self, rename: Callable[[str], None], delete: Callable[[], None]
    ) -> None:
        super().__init__("div")
        self.classes("edit-actions row-line")
        self.rename = rename
        self.on_delete = delete
        self.confirmed = False
        with self:
            self.editor = ui.element("input").props("placeholder=name aria-label=name")
            self.editor.set_visibility(False)
            self.editor.on(
                "keydown.enter", self.submit, js_handler="e => emit(e.target.value)"
            )
            self.rename_button = Button("rename", self.edit)
            self.delete_button = Button("delete", self.request_delete)

    def edit(self) -> None:
        self.rename_button.set_visibility(False)
        self.editor.set_visibility(True)
        self.editor.run_method("focus")

    def submit(self, event: GenericEventArguments) -> None:
        name = event.args.strip()
        if name:
            self.rename(name)
            self.editor.set_visibility(False)
            self.rename_button.set_visibility(True)

    def request_delete(self) -> None:
        if self.confirmed:
            self.on_delete()
        else:
            self.confirmed = True
            self.delete_button.set_text("confirm")
