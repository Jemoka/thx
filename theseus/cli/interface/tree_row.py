"""A retained terminal tree row with a right-aligned modification time."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theseus.cli.interface.tree import RunTree

from functools import partial

from nicegui import ui

from theseus.cli.interface.age import AgeLabel
from theseus.cli.interface.navigation import Button


class TreeRow(Button):
    def __init__(self, tree: RunTree, path: tuple[str, ...]) -> None:
        super().__init__("", partial(tree.activate, path), classes="tree-row")
        self.path = path
        self.style(f"padding-left:{len(path) * 2}ch")
        self.props.update({"role": "treeitem", "aria-level": len(path) + 1})
        self.on("focus", partial(tree.select, path))
        self.on("mouseenter", partial(tree.hover, path))
        self.on("mouseleave", lambda: setattr(tree, "hovered", None))
        with self:
            self.mark_label = ui.label("").classes("tree-mark")
            self.age = AgeLabel()

    def set(
        self,
        label: str,
        modified: int | None,
        expanded: bool,
        marked: bool,
        selected: bool,
    ) -> None:
        branch = len(self.path) < 4
        glyph = ("▼ " if expanded else "▶ ") if branch else "  "
        self.set_text(glyph + label)
        self.mark_label.set_text("  ●" if marked else "")
        self.props["aria-selected"] = selected
        if branch:
            self.props["aria-expanded"] = expanded
        self.age.set(modified)
