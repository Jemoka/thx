"""Live run hierarchy with retained expansion, cursor, and descendant marks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nicegui.events import GenericEventArguments

from theseus.cli.interface.navigation import KeyEvent

if TYPE_CHECKING:
    from theseus.cli.interface.driver import TheseusInterface

import re

from nicegui import ui

from theseus.cli.interface.data import RunKey
from theseus.cli.interface.navigation import Screen
from theseus.cli.interface.tree_row import TreeRow
from theseus.cli.interface.views import ViewList


class RunTree(Screen):
    def __init__(self, app: TheseusInterface) -> None:
        super().__init__(app)
        self.search = ""
        self.expanded: set[tuple[str, ...]] = {()}
        self.cursor: tuple[str, ...] = ()
        self.marked: set[RunKey] = set()
        self.entries: dict[tuple[str, ...], list[tuple[RunKey, int]]] = {}
        self.hovered: tuple[str, ...] | None = None
        self.buttons: dict[tuple[str, ...], TreeRow] = {}
        with self:
            self.search_input = (
                ui.element("input")
                .props("placeholder=search aria-label=search")
                .classes("run-search")
            )
            self.search_input.on(
                "input", self.filter, js_handler="e=>emit(e.target.value)"
            )
            self.search_input.on(
                "keydown.down.prevent", lambda: self.focus(self.cursor)
            )
            self.tree = (
                ui.element("div").classes("run-tree scroll key-list").props("role=tree")
            )
            self.workspace_pane = ui.element("div").classes("workspace")
            with self.workspace_pane:
                self.marked_label = ui.label("").classes("marked-runs")
                ui.label("cross-cutting views").classes("bold")
                self.views = ViewList(app)
            self.workspace_pane.set_visibility(False)
        self.refresh_data()

    def filter(self, event: GenericEventArguments) -> None:
        self.search = event.args
        self.search_input.props["value"] = self.search
        self.refresh_data()
        self.app.update_location(replace=True)

    def refresh_data(self) -> None:
        try:
            pattern = re.compile(self.search)
        except re.error:
            pattern = None
        entries: dict[tuple[str, ...], list[tuple[RunKey, int]]] = {(): []}
        for (name, nonce), modified in self.app.runs.items():
            project, group, leaf = name.split(".", 2)
            if pattern and not any(pattern.search(v) for v in (project, group, nonce)):
                continue
            path: tuple[str, ...] = (project, group, leaf, nonce)
            for depth in range(5):
                entries.setdefault(path[:depth], []).append(
                    (RunKey(name, nonce), modified)
                )
        self.entries = entries
        children: dict[tuple[str, ...], list[tuple[str, ...]]] = {
            path: [] for path in entries
        }
        for path in entries:
            if path:
                children[path[:-1]].append(path)
        for path, items in children.items():
            items.sort(
                key=lambda child: (
                    (-max(t for _, t in entries[child]), child)
                    if len(child) == 2
                    else (0, child)
                )
            )
        visible = []
        stack: list[tuple[str, ...]] = [()]
        while stack:
            path = stack.pop()
            visible.append(path)
            if path in self.expanded:
                stack.extend(reversed(children[path]))
        self.visible = visible
        if self.cursor not in visible:
            self.cursor = ()
        for path in set(self.buttons) - set(visible):
            self.buttons.pop(path).delete()
        with self.tree:
            for index, path in enumerate(visible):
                if path not in self.buttons:
                    self.buttons[path] = TreeRow(self, path)
                row = self.buttons[path]
                row.move(self.tree, target_index=index)
                runs = entries[path]
                row.set(
                    path[-1] if path else str(self.app.root_path),
                    max((t for _, t in runs), default=None),
                    path in self.expanded,
                    len(path) == 4 and runs[0][0] in self.marked,
                    path == self.cursor,
                )

    def select(self, path: tuple[str, ...]) -> None:
        self.cursor = path
        for row, button in self.buttons.items():
            button.props["aria-selected"] = row == path

    def hover(self, path: tuple[str, ...]) -> None:
        self.hovered = path

    def focus(self, path: tuple[str, ...]) -> None:
        self.cursor = path
        self.buttons[path].run_method("focus")

    async def activate(self, path: tuple[str, ...]) -> None:
        self.cursor = path
        if len(path) == 4:
            await self.app.open_run(self.entries[path][0][0])
        else:
            self.expanded.symmetric_difference_update({path})
            self.refresh_data()
            self.focus(path)
            self.app.update_location()

    def refresh_marks(self) -> None:
        self.workspace_pane.set_visibility(bool(self.marked))
        self.marked_label.set_text(
            "marked runs\n"
            + "\n".join(
                f"{r.name}  {r.nonce}"
                for r in sorted(self.marked, key=lambda r: (r.name, r.nonce))
            )
        )

    def key(self, event: KeyEvent) -> None:
        key = event["key"]
        if key in ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"):
            self.hovered = None
        if key in ("m", "u"):
            path = getattr(self, "hovered", None) or self.cursor
            runs = {run for run, _ in self.entries.get(path, ())}
            if key == "m":
                self.marked.update(runs)
            else:
                self.marked.difference_update(runs)
            self.refresh_marks()
        elif key == "ArrowLeft":
            if self.cursor in self.expanded:
                self.expanded.remove(self.cursor)
            elif self.cursor:
                self.cursor = self.cursor[:-1]
        elif key == "ArrowRight":
            index = self.visible.index(self.cursor)
            if self.cursor in self.expanded and index + 1 < len(self.visible):
                self.focus(self.visible[index + 1])
                return
            self.expanded.add(self.cursor)
        elif key in ("ArrowUp", "ArrowDown", "Home", "End"):
            index = self.visible.index(self.cursor)
            if key == "ArrowUp" and index == 0:
                self.search_input.run_method("focus")
                return
            index = (
                0
                if key == "Home"
                else len(self.visible) - 1
                if key == "End"
                else min(
                    len(self.visible) - 1,
                    max(0, index + (-1 if key == "ArrowUp" else 1)),
                )
            )
            self.focus(self.visible[index])
            return
        else:
            return
        self.refresh_data()
        self.focus(self.cursor)
        self.app.update_location()

    def resume(self) -> None:
        super().resume()
        self.views.refresh()
