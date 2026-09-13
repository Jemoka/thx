"""Run status inferred from ranked bootstrap logs."""

from dataclasses import dataclass
from pathlib import Path
import re
from time import time_ns

from nicegui import run, ui

from theseus.cli.interface.data import RunData, RunKey
from theseus.cli.interface.log_data import LogFile


@dataclass(frozen=True)
class LogStatus:
    """A best-effort observation; quiet logs do not prove a process exited."""

    state: str
    modified: int = 0

    @classmethod
    def read(cls, directory: Path, data: RunData) -> "LogStatus":
        try:
            files = LogFile.discover(
                directory, RunKey(data.name, data.nonce), data.executions[-1:]
            )
            states = []
            for file in files:
                with file.path.open("rb") as source:
                    source.seek(0, 2)
                    source.seek(max(0, source.tell() - 65536))
                    tail = source.read().decode("utf-8", errors="replace")
                tail = tail.rsplit("[bootstrap] dispatch nonce=", 1)[-1]
                exits = re.findall(
                    r"\[bootstrap\] cleanup started with exit code (\d+)", tail
                )
                if exits:
                    states.append("completed" if int(exits[-1]) == 0 else "failed")
                elif time_ns() - file.modified > 300_000_000_000:
                    states.append("stale")
                else:
                    states.append("running")
        except OSError:
            return cls("unavailable")
        if not states:
            return cls("no logs")
        state = next(
            (
                candidate
                for candidate in ("failed", "stale", "running")
                if candidate in states
            ),
            "completed",
        )
        return cls(state, max(file.modified for file in files))


class RunStatus(ui.label):
    """Refresh independently of object-store writes, including on quiet runs."""

    def __init__(self, directory: Path | None, data: RunData) -> None:
        super().__init__(
            "status unavailable" if directory is None else "checking status"
        )
        self.directory, self.data = directory, data
        self.classes("run-status muted")
        self.props("role=status aria-live=polite")
        self.tooltip(
            "Status inferred from bootstrap logs; stale means no log updates for five minutes."
        )
        if directory is not None:
            ui.timer(5, self.refresh)

    async def refresh(self) -> None:
        if self.directory is None:
            return
        status = await run.io_bound(LogStatus.read, self.directory, self.data)
        if status is not None:
            self.set_text(status.state)
            self.props["data-status"] = status.state
