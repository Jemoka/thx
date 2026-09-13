"""Run data prepared for the interactive interface."""

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from math import ceil
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from theseus.base import Node
from theseus.cli.interface.plot_math import interpolate
from theseus.store import ObjectReader, ValueRow

if TYPE_CHECKING:
    from pandas import DataFrame


@dataclass(frozen=True)
class TimelineData:
    """A fixed-resolution summary of one run's steps."""

    first: int
    last: int
    steps: tuple[int, ...]
    bins: tuple[int, ...]


@dataclass(frozen=True)
class RunKey:
    """Identify one stored run independently of its sequence."""

    name: str
    nonce: str


@dataclass(frozen=True)
class PlotSeries:
    """Name one scalar series rendered by a plot."""

    name: str
    run: RunKey
    points: tuple[tuple[float, float], ...]
    sequences: tuple[tuple[float, int], ...]
    checkpoints: tuple[tuple[float, float], ...] = ()
    checkpoint_sequences: tuple[int, ...] = ()


@dataclass(frozen=True)
class RunSnapshot:
    """Values and checkpoints read for one run at a single instant."""

    rows: tuple[tuple[Node, ValueRow], ...]
    checkpoints: frozenset[int]
    executions: tuple[str, ...] = ()


class DataFrameSource(Protocol):
    """Provide complete tabular data for external exploration."""

    def to_dataframe(self) -> "DataFrame":
        """Return every source row in a tabular representation."""
        ...


class PlotData(DataFrameSource, Protocol):
    """Provide the fields and series consumed by saved plots."""

    @property
    def comparison(self) -> bool: ...

    @property
    def scalar_keys(self) -> tuple[str, ...]: ...

    def plot_series(self, x_key: str, y_key: str, limit: int) -> tuple[PlotSeries, ...]:
        """Return bounded named series for the selected fields."""
        ...


class RunData:
    """Materialize one run once and expose UI-shaped projections of it."""

    def __init__(
        self,
        store: ObjectReader,
        name: str,
        nonce: str,
    ) -> None:
        self.name = name
        self.nonce = nonce
        self._snapshot = RunSnapshot((), frozenset())
        self.rows: dict[int, tuple[Node, ValueRow]] = {}
        self.steps: tuple[int, ...] = ()
        self.checkpoints: frozenset[int] = frozenset()
        self.executions: tuple[str, ...] = ()
        self.apply(self.read(store))

    def read(self, store: ObjectReader) -> RunSnapshot:
        """Read a coherent replacement snapshot without mutating this run."""
        raw_rows = (
            store.query()
            .name(self.name)
            .nonce(self.nonce)
            .select(return_nodes=True, raw=True)
        )
        rows: list[tuple[Node, ValueRow]] = []
        executions: dict[str, int] = {}
        for node, raw_values in raw_rows:
            execution = raw_values.get("_x_execution")
            written = raw_values.get("_x_write", 0)
            if (
                isinstance(execution, str)
                and execution
                and isinstance(written, (int, float))
            ):
                executions[execution] = max(executions.get(execution, 0), int(written))
            values = {
                key: value
                for key, value in raw_values.items()
                if not key.startswith("_x_")
            }
            timestamp = raw_values.get("_x_timestamp")
            if isinstance(timestamp, Real):
                values["timestamp"] = timestamp / 1_000_000_000
            rows.append((node, values))
        checkpoints = frozenset(
            node.seq
            for node in (
                store.query().name(self.name).nonce(self.nonce).checkpoint().all()
            )
        )
        return RunSnapshot(
            tuple(rows),
            checkpoints,
            tuple(sorted(executions, key=lambda key: executions[key])),
        )

    def apply(self, snapshot: RunSnapshot) -> bool:
        """Apply a snapshot and invalidate cached scalar projections if changed."""
        if snapshot == self._snapshot:
            return False
        self._snapshot = snapshot
        self.rows = {node.seq: (node, values) for node, values in snapshot.rows}
        self.steps = tuple(self.rows)
        self.checkpoints = snapshot.checkpoints
        self.executions = snapshot.executions
        self.series.cache_clear()
        return True

    @property
    def comparison(self) -> bool:
        """Return whether plots contain multiple runs."""
        return False

    @property
    def scalar_keys(self) -> tuple[str, ...]:
        """Return numeric user fields available anywhere in the run."""
        return tuple(
            sorted(
                {
                    key
                    for _, values in self.rows.values()
                    for key, value in values.items()
                    if isinstance(value, Real)
                }
            )
        )

    def timeline(self, width: int) -> TimelineData:
        """Collapse run events into at most ``width`` timeline bins."""
        if not self.steps:
            return TimelineData(0, 0, (), ())

        first, last = self.steps[0], self.steps[-1]
        count = min(max(1, width), last - first + 1)
        bins = [0] * count
        span = max(1, last - first)
        for seq, (_, values) in self.rows.items():
            index = (seq - first) * (count - 1) // span
            if values:
                bins[index] |= 1
            if seq in self.checkpoints:
                bins[index] |= 2
        return TimelineData(first, last, self.steps, tuple(bins))

    def nearest(self, seq: int) -> int:
        """Return the recorded step nearest to ``seq``."""
        if not self.steps:
            return 0
        index = bisect_left(self.steps, seq)
        if index == 0:
            return self.steps[0]
        if index == len(self.steps):
            return self.steps[-1]
        before, after = self.steps[index - 1 : index + 1]
        return before if seq - before <= after - seq else after

    def step(self, seq: int) -> tuple[Node, ValueRow]:
        """Return the identity and folded fields for an exact recorded step."""
        return self.rows[self.nearest(seq)]

    @lru_cache(maxsize=32)
    def series(
        self, x_key: str, y_key: str, limit: int
    ) -> tuple[tuple[float, float], ...]:
        """Return a bounded scalar series, preserving its final point."""
        points: list[tuple[float, float]] = []
        for seq, (_, values) in self.rows.items():
            x = seq if x_key == "step" else values.get(x_key)
            y = seq if y_key == "step" else values.get(y_key)
            if isinstance(x, Real) and isinstance(y, Real):
                points.append((float(x), float(y)))

        stride = max(1, ceil(len(points) / max(1, limit)))
        sampled = points[::stride]
        if points and sampled[-1] != points[-1]:
            sampled.append(points[-1])
        return tuple(sampled)

    def plot_series(self, x_key: str, y_key: str, limit: int) -> tuple[PlotSeries, ...]:
        """Return this run as one named plot series."""
        points = self.series(x_key, y_key, limit)
        if not points:
            return ()
        seq_x: list[tuple[float, float]] = []
        for seq, (_, values) in self.rows.items():
            value = seq if x_key == "step" else values.get(x_key)
            if isinstance(value, Real):
                seq_x.append((float(seq), float(value)))
        checkpoints: list[tuple[float, float]] = []
        checkpoint_sequences: list[int] = []
        for seq in sorted(self.checkpoints):
            x = interpolate(seq_x, float(seq))
            y = interpolate(sorted(points), x) if x is not None else None
            if x is not None and y is not None:
                checkpoints.append((x, y))
                checkpoint_sequences.append(seq)
        return (
            PlotSeries(
                self.nonce,
                RunKey(self.name, self.nonce),
                points,
                tuple((x, int(seq)) for seq, x in seq_x),
                tuple(checkpoints),
                tuple(checkpoint_sequences),
            ),
        )

    def to_dataframe(self) -> "DataFrame":
        """Return every run log row with its node identity and checkpoint state."""
        from pandas import DataFrame

        return DataFrame.from_records(self._records())

    def _records(self) -> list[dict[str, object]]:
        return [
            {
                **{
                    key: (
                        datetime.fromtimestamp(float(value))
                        if key == "timestamp" and isinstance(value, Real)
                        else str(value)
                        if isinstance(value, Path)
                        else value
                    )
                    for key, value in values.items()
                },
                "_x_name": node.name,
                "_x_nonce": node.nonce,
                "_x_seq": node.seq,
                "_x_checkpoint": node.seq in self.checkpoints,
            }
            for node, values in self.rows.values()
        ]


class WorkspaceData:
    """Expose comparable scalar series across a marked set of runs."""

    def __init__(self, runs: tuple[RunData, ...]) -> None:
        self.runs = runs

    @property
    def comparison(self) -> bool:
        """Return whether plots belong to a cross-cutting workspace."""
        return True

    @property
    def scalar_keys(self) -> tuple[str, ...]:
        """Return numeric fields available in every currently marked run."""
        if not self.runs:
            return ()
        keys = set(self.runs[0].scalar_keys)
        for run in self.runs[1:]:
            keys.intersection_update(run.scalar_keys)
        return tuple(sorted(keys))

    def plot_series(self, x_key: str, y_key: str, limit: int) -> tuple[PlotSeries, ...]:
        """Return one line for each marked run containing both fields."""
        series: list[PlotSeries] = []
        for run in self.runs:
            source = run.plot_series(x_key, y_key, limit)
            if source:
                series.append(
                    PlotSeries(
                        f"{run.name.rsplit('.', 1)[-1]} {run.nonce}",
                        source[0].run,
                        source[0].points,
                        source[0].sequences,
                        source[0].checkpoints,
                        source[0].checkpoint_sequences,
                    )
                )
        return tuple(series)

    def to_dataframe(self) -> "DataFrame":
        """Return every log row from every run in this workspace."""
        from pandas import DataFrame

        return DataFrame.from_records(
            [record for run in self.runs for record in run._records()]
        )
