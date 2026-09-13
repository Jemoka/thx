"""Bookmarkable interface locations, independent of browser navigation state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from theseus.cli.interface.data import RunData, WorkspaceData
from theseus.cli.interface.state import View

if TYPE_CHECKING:
    from theseus.cli.interface.navigation import Screen
    from theseus.cli.interface.tree import RunTree

import json
from urllib.parse import parse_qs, urlencode

from theseus.cli.interface.data import RunKey


def view_location(data: RunData | WorkspaceData, view: View) -> dict[str, str]:
    """Describe a saved view and its optional single-run scope."""
    location = {"screen": "view", "view": str(view.id)}
    if isinstance(data, RunData):
        location.update(run=data.name, nonce=data.nonce)
    return location


class Location:
    """Encode navigation and marked runs without changing the SQLite schema."""

    def __init__(self, query: str = "") -> None:
        self.values = {
            key: values[-1] for key, values in parse_qs(query.lstrip("?")).items()
        }

    @property
    def marked(self) -> set[RunKey]:
        return {RunKey(*pair) for pair in json.loads(self.values.get("marked", "[]"))}

    @property
    def expanded(self) -> set[tuple[str, ...]]:
        return {tuple(path) for path in json.loads(self.values.get("expanded", "[[]]"))}

    def url(self, screen: Screen, home: RunTree) -> str:
        values = {**screen.location}
        if home.marked:
            values["marked"] = json.dumps(
                [
                    (key.name, key.nonce)
                    for key in sorted(
                        home.marked, key=lambda key: (key.name, key.nonce)
                    )
                ],
                separators=(",", ":"),
            )
        if home.expanded != {()}:
            values["expanded"] = json.dumps(
                sorted(home.expanded), separators=(",", ":")
            )
        if home.search:
            values["search"] = home.search
        query = urlencode(values)
        return "/" + ("?" + query if query else "")
