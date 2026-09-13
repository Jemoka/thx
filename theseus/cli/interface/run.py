"""Run detail screen with timeline, fields, and saved views."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui.events import GenericEventArguments

from theseus.base import Node
from theseus.cli.interface.data import RunData
from theseus.cli.interface.navigation import KeyEvent
from theseus.store import ValueRow

if TYPE_CHECKING:
    from theseus.cli.interface.driver import TheseusInterface

from nicegui import run, ui

from theseus.cli.interface.fields import NodeIdentity, StepFields
from theseus.cli.interface.status import RunStatus
from theseus.cli.interface.navigation import Button, Screen
from theseus.cli.interface.timeline import RunTimeline
from theseus.cli.interface.views import ViewList
from theseus.cli.interface.walk import WalkButton


class RunScreen(Screen):
    def __init__(
        self, app: TheseusInterface, data: RunData, seq: int | None = None
    ) -> None:
        super().__init__(app)
        self.data = data
        self.displayed_step: tuple[Node, ValueRow] | None = None
        with self:
            with ui.element("div").classes("run-top"):
                with ui.element("div").classes("row-line"):
                    Button("‹", app.pop_screen)
                    ui.label(f"{data.name}  {data.nonce}").classes("grow")
                    self.status = RunStatus(app.logs_path, data)
                    self.jump_input = (
                        ui.element("input")
                        .props("placeholder=seq aria-label=seq inputmode=numeric")
                        .classes("jump-input")
                    )
                    self.jump_input.set_visibility(False)
                    self.jump_input.on(
                        "keydown.enter",
                        self.jump,
                        js_handler="e => emit(e.target.value)",
                    )
                    if app.logs_path is not None:
                        Button("logs", partial(app.open_logs, data))
                    WalkButton(data)
                    self.jump_button = Button("jump", self.edit_jump)
                self.timeline = RunTimeline(data, self.show_step, seq)
                with ui.element("div").classes("row-line"):
                    self.step_label = ui.label("")
                    self.checkpoint_label = Button(
                        "checkpoint", self.open_checkpoint, classes="checkpoint-label"
                    )
                self.identity = NodeIdentity()
                self.fields = StepFields()
            self.views = ViewList(app, data)
        self.show_step(self.timeline.selected)
        self.timeline.run_method("focus")

    @property
    def location(self) -> dict[str, str]:
        return {
            "screen": "run",
            "run": self.data.name,
            "nonce": self.data.nonce,
            "seq": str(self.timeline.selected),
        }

    def show_step(self, seq: int) -> None:
        if not self.data.steps:
            self.displayed_step = None
            self.checkpoint_label.set_visibility(False)
            self.step_label.set_text("no recorded steps")
            self.identity.set("")
            self.fields.set({})
            return
        node, values = self.data.step(seq)
        self.step_label.set_text(f"seq {node.seq}")
        self.checkpoint_label.set_visibility(node.seq in self.data.checkpoints)
        self.identity.set(node.serialize())
        if self.app.stack and self.app.stack[-1] is self:
            self.app.update_location(replace=True)
        if self.displayed_step != (node, values):
            self.fields.set(values)
            self.displayed_step = (node, values)

    async def open_checkpoint(self) -> None:
        if self.displayed_step is None:
            return
        node, values = self.displayed_step
        blob = values.get("blob")
        if not isinstance(blob, (Path, str)):
            ui.notify("Checkpoint files are unavailable", type="warning")
            return
        path = Path(blob).resolve()
        if not path.is_relative_to((self.app.cache.reader.root() / "blobs").resolve()):
            ui.notify("Checkpoint is outside this store", type="negative")
            return
        documents = {}
        for name in ("config.yaml", "job.json"):
            try:
                documents[name] = await run.io_bound((path / name).read_text)
            except OSError:
                documents[name] = "unavailable"
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl"):
            ui.label(f"Checkpoint · seq {node.seq}")
            for name, content in documents.items():
                ui.label(name)
                ui.code(
                    content or "", language="yaml" if name.endswith("yaml") else "json"
                ).classes("w-full")
            Button("close", dialog.close)
        dialog.open()

    def edit_jump(self) -> None:
        self.jump_button.set_visibility(False)
        self.jump_input.set_visibility(True)
        self.jump_input.run_method("focus")

    def jump(self, event: GenericEventArguments) -> None:
        if event.args:
            try:
                self.timeline.select(int(event.args))
            except ValueError:
                return
        self.jump_input.set_visibility(False)
        self.jump_button.set_visibility(True)
        self.timeline.run_method("focus")

    def key(self, event: KeyEvent) -> None:
        key = event["key"]
        if key in ("ArrowLeft", "h", "ArrowRight", "l"):
            self.timeline.scrub(
                -1 if key in ("ArrowLeft", "h") else 1, event["shiftKey"]
            )
        elif key in ("PageUp", "PageDown"):
            self.timeline.scrub(
                max(1, len(self.data.steps) // 20) * (-1 if key == "PageUp" else 1)
            )
        elif key in ("Home", "End") and self.data.steps:
            self.timeline.select(self.data.steps[0 if key == "Home" else -1])
        else:
            super().key(event)

    def refresh_data(self) -> None:
        self.timeline.refresh()
        self.show_step(self.timeline.selected)

    def resume(self) -> None:
        super().resume()
        self.views.refresh()
