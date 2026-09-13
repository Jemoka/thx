"""Discovery and incremental reading of new-execute bootstrap logs."""

from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from theseus.cli.interface.data import RunKey


_ANSI_ESCAPE = re.compile(rb"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")


@dataclass(frozen=True)
class LogFile:
    """One ranked bootstrap log relevant to a stored run."""

    path: Path
    rank: int
    modified: int

    @classmethod
    def discover(
        cls, directory: Path, run: RunKey, executions: tuple[str, ...] = ()
    ) -> tuple[LogFile, ...]:
        """Return this run's new-execute log files in machine-rank order."""
        project, group, name = run.name.split(".", 2)
        stem = "-".join(
            part.replace("/", "_") for part in (project, group, name, run.nonce)
        )
        prefix, suffix = f"{stem}.", ".log"
        files: list[LogFile] = []
        with os.scandir(directory) as entries:
            for entry in entries:
                match = re.fullmatch(r"(.+)\.(\d+)\.log", entry.name)
                if executions:
                    if match is None or not any(
                        match[1].endswith("-" + execution.replace("/", "_"))
                        for execution in executions
                    ):
                        continue
                    rank = match[2]
                else:
                    if not entry.name.startswith(prefix) or not entry.name.endswith(
                        suffix
                    ):
                        continue
                    rank = entry.name[len(prefix) : -len(suffix)]
                if not rank.isdigit() or not entry.is_file():
                    continue
                try:
                    stat = entry.stat()
                except FileNotFoundError:
                    continue
                files.append(cls(Path(entry.path), int(rank), stat.st_mtime_ns))
        return tuple(sorted(files, key=lambda file: (file.rank, file.path.name)))


@dataclass(frozen=True)
class LogUpdate:
    """Complete and partial lines observed during one log read."""

    reset: bool
    lines: tuple[tuple[int, str], ...]
    partial: tuple[int, str] | None
    modified: int
    size: int
    total_lines: int


class LogReader:
    """Read an exact, bounded tail once and then follow appended lines."""

    def __init__(self, path: Path, limit: int) -> None:
        self.path = path
        self.limit = max(1, limit)
        self.identity: tuple[int, int] | None = None
        self.offset = 0
        self.next_line = 1
        self.pending = b""

    def read(self) -> LogUpdate:
        """Read the latest tail or the bytes appended since the previous call."""
        with self.path.open("rb") as source:
            stat = os.fstat(source.fileno())
            identity = (stat.st_dev, stat.st_ino)
            reset = identity != self.identity or stat.st_size < self.offset
            self.identity = identity
            if reset:
                return self._read_latest(source, stat.st_mtime_ns, stat.st_size)
            return self._read_appended(source, stat.st_mtime_ns, stat.st_size)

    def _read_latest(self, source: BinaryIO, modified: int, size: int) -> LogUpdate:
        lines: deque[tuple[int, str]] = deque(maxlen=self.limit)
        self.next_line = 1
        self.pending = b""
        while raw := source.readline():
            if raw.endswith(b"\n"):
                lines.append((self.next_line, self._decode(raw[:-1])))
                self.next_line += 1
            else:
                self.pending = raw
        self.offset = source.tell()
        return self._update(True, tuple(lines), modified, size)

    def _read_appended(self, source: BinaryIO, modified: int, size: int) -> LogUpdate:
        source.seek(self.offset)
        content = self.pending + source.read()
        self.offset = source.tell()
        raw_lines = content.split(b"\n")
        self.pending = raw_lines.pop() if raw_lines else b""
        count = len(raw_lines)
        first = self.next_line + max(0, count - self.limit)
        lines = tuple(
            (number, self._decode(raw))
            for number, raw in enumerate(raw_lines[-self.limit :], first)
        )
        self.next_line += count
        return self._update(False, lines, modified, size)

    def _update(
        self,
        reset: bool,
        lines: tuple[tuple[int, str], ...],
        modified: int,
        size: int,
    ) -> LogUpdate:
        partial = (self.next_line, self._decode(self.pending)) if self.pending else None
        return LogUpdate(
            reset,
            lines,
            partial,
            modified,
            size,
            self.next_line if partial is not None else self.next_line - 1,
        )

    @staticmethod
    def _decode(raw: bytes) -> str:
        return (
            _ANSI_ESCAPE.sub(b"", raw)
            .decode("utf-8", errors="replace")
            .rstrip("\r")
            .replace("\x00", "�")
        )
