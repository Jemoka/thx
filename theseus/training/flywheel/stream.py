"""Deterministic sample plans and sequential, node-clocked prefetching.

A plan fixes both mixture membership and sample locations before any batch is
read. Plans are independently seeded, so restoring a sequence skips directly to
its plan without replaying previous batches. Workers only advance a private
cursor; the training node controls which prefetched batch is visible.
"""

from collections.abc import Iterator, Sequence
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, TYPE_CHECKING

import jax
import numpy as np

from theseus.base import Node

if TYPE_CHECKING:
    from .dataset import Dataset

PLAN_SAMPLES = 65536


def batches(
    datasets: Sequence["Dataset"],
    rates: Sequence[float],
    batch_size: int,
    split: str = "train",
    seed: int = 0,
    start: int = 0,
    validate: bool = True,
) -> Iterator[dict[str, np.ndarray]]:
    """Read a host slice of a shared, rank- and nonce-independent sample plan.

    Sequence s covers global positions [s * G, (s + 1) * G), where
    G = batch_size * process_count. Host r owns its contiguous batch_size
    positions. Keeping G, seed, and dataset contents fixed preserves the samples
    when host count changes; sampling itself can still repeat dataset rows.
    """
    if batch_size <= 0 or start < 0:
        raise ValueError("Batch size must be positive and sequence nonnegative")
    plan_size = max(
        PLAN_SAMPLES,
        *(
            min(dataset._size(split), getattr(dataset, "_window_rows", 0))
            for dataset in datasets
        ),
    )
    rank, hosts = jax.process_index(), jax.process_count()
    position = (start * hosts + rank) * batch_size
    plan_number = -1
    membership = np.empty(0, dtype=np.int64)
    location_segments: list[tuple[int, np.ndarray] | None] = [None] * len(datasets)
    while True:
        parts: list[dict[str, np.ndarray]] = []
        cursor = position
        remaining = batch_size
        while remaining:
            number, offset = divmod(cursor, plan_size)
            if number != plan_number:
                rng = np.random.default_rng([seed, int(split != "train"), number])
                membership = (
                    rng.choice(len(datasets), plan_size, p=rates)
                    if len(datasets) > 1
                    else np.zeros(plan_size, dtype=np.int8)
                )
                plan_number = number
            count = min(remaining, plan_size - offset)
            chosen = membership[offset : offset + count]
            part: dict[str, np.ndarray] = {}
            for index, dataset in enumerate(datasets):
                slots = np.flatnonzero(chosen == index) if len(datasets) > 1 else None
                if slots is not None and not len(slots):
                    continue
                planned = np.empty(count, dtype=np.int64)
                planned_position = cursor
                planned_count = 0
                while planned_count < count:
                    cached = location_segments[index]
                    if cached is None or not (
                        cached[0] <= planned_position < cached[0] + len(cached[1])
                    ):
                        cached = dataset._plan_segment(
                            split,
                            seed,
                            index,
                            planned_position,
                            plan_size,
                        )
                        location_segments[index] = cached
                    segment_start, segment = cached
                    segment_offset = planned_position - segment_start
                    take = min(count - planned_count, len(segment) - segment_offset)
                    planned[planned_count : planned_count + take] = segment[
                        segment_offset : segment_offset + take
                    ]
                    planned_position += take
                    planned_count += take
                indices = planned if slots is None else planned[slots]
                values = dataset._read(
                    indices,
                    split,
                    (seed, int(split != "train"), cursor, index) if validate else None,
                    offsets=slots,
                )
                if len(datasets) == 1:
                    part = values
                    break
                for key, value in values.items():
                    if key not in part:
                        part[key] = np.empty(
                            (count, *value.shape[1:]), dtype=value.dtype
                        )
                    part[key][slots] = value
            parts.append(part)
            cursor += count
            remaining -= count
        yield (
            parts[0]
            if len(parts) == 1
            else {
                key: np.concatenate([part[key] for part in parts]) for key in parts[0]
            }
        )
        position += batch_size * hosts


class AsyncStrategy:
    """Cache the current node's batch while prefetching future steps in order.

    Without a node, each read advances an owned clock for legacy sequential
    callers. An explicitly supplied node is advanced only by its owner.
    """

    def __init__(
        self,
        datasets: Sequence["Dataset"],
        rates: Sequence[float],
        batch_size: int,
        split: str = "train",
        node: Node | None = None,
        seed: int = 0,
    ) -> None:
        self.node = node if node is not None else Node(name="loader")
        self._owned = node is None
        self._args = (datasets, rates, batch_size, split, seed)
        self.queue: Queue[Any] = Queue(maxsize=8)
        self._stop = Event()
        self.thread: Thread | None = None
        self._sequence = -1
        self._batch: dict[str, np.ndarray] | None = None
        self.error: Exception | None = None

    def get_batch(self) -> dict[str, np.ndarray]:
        if self.error is not None:
            raise self.error
        if self._stop.is_set():
            raise RuntimeError("Loader is closed")
        sequence = self.node.seq
        if sequence < self._sequence:
            raise ValueError("A live loader cannot rewind; construct a new loader")
        if self.thread is None:
            self.thread = Thread(
                target=self._fetch_worker, args=(sequence,), daemon=True
            )
            self.thread.start()
        while self._sequence < sequence:
            item = self.queue.get()
            if isinstance(item, Exception):
                self.error = item
                raise item
            self._sequence, self._batch = item
        assert self._batch is not None
        if self._owned:
            self.node.update(self.node.next())
        return self._batch

    def _fetch_worker(self, start: int) -> None:
        stream = batches(*self._args, start=start)
        sequence = start
        while not self._stop.is_set():
            try:
                item: Any = (sequence, next(stream))
            except Exception as error:
                item = error
            while not self._stop.is_set():
                try:
                    self.queue.put(item, timeout=0.1)
                    break
                except Full:
                    continue
            if isinstance(item, Exception):
                return
            sequence += 1

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        if self.thread is not None:
            self.thread.join()
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break

    def __del__(self) -> None:
        self.close()
