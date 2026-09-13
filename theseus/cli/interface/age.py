"""Relative modification-time labels shared by terminal lists."""

from datetime import datetime
from time import time_ns

from nicegui.elements.label import Label


class AgeLabel(Label):
    """Render a nanosecond timestamp using the home tree's compact age format."""

    def __init__(self, modified: int | None = None) -> None:
        super().__init__()
        self.classes("age")
        self.set(modified)

    def set(self, modified: int | None) -> None:
        if modified is None:
            self.set_text("")
            return
        seconds = max(0, (time_ns() - modified) // 1_000_000_000)
        if seconds < 60:
            age = "now"
        elif seconds < 3600:
            age = f"{seconds // 60}m ago"
        elif seconds < 86400:
            age = f"{seconds // 3600}h ago"
        elif seconds < 604800:
            age = f"{seconds // 86400}d ago"
        else:
            date = datetime.fromtimestamp(modified / 1e9)
            age = f"{date:%b} {date.day}"
            if date.year != datetime.now().year:
                age += f", {date.year}"
        self.set_text(age)
