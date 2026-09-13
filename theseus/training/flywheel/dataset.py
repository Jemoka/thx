"""Reader contract: plan sample locations, then read the requested rows."""

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

from .stream import batches


class Dataset(ABC):
    @abstractmethod
    def _size(self, split: str) -> int: ...

    @abstractmethod
    def _read_rows(self, indices: np.ndarray, split: str) -> dict[str, np.ndarray]: ...

    def _plan(self, split: str, rng: np.random.Generator, count: int) -> np.ndarray:
        return rng.integers(self._size(split), size=count)

    def _read(
        self,
        indices: np.ndarray,
        split: str,
        seed: tuple[int, int, int, int] | None = None,
        offsets: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Replace empty rows using global stream positions, independent of hosts.

        Offsets identify mixture slots within the chunk; an unmixed chunk uses
        each row's position directly. Valid batches keep the bulk read path.
        """
        values = self._read_rows(indices, split)
        if seed is not None:
            for attempt in range(11):
                x, y = (
                    (values["pos"], values["neg"])
                    if "pos" in values
                    else (values["x"], values["y"])
                )
                axes = tuple(range(1, x.ndim))
                invalid = ~x.any(axis=axes) & ~y.any(axis=axes)
                if not invalid.any():
                    break
                if attempt == 10:
                    raise ValueError("Dataset repeatedly returned all-zero samples")
                data_seed, split_seed, start, dataset_index = seed
                rows = np.flatnonzero(invalid)
                indices = np.empty(len(rows), dtype=np.int64)
                for slot, row in enumerate(rows):
                    position = start + int(row if offsets is None else offsets[row])
                    rng = np.random.default_rng(
                        [data_seed, split_seed, position, dataset_index, attempt]
                    )
                    indices[slot] = self._plan(split, rng, 1)[0]
                # PMD reads must stay within one window. Group retries there;
                # padded readers can fetch all replacements concurrently.
                window_rows = getattr(self, "_window_rows", 0)
                groups = (
                    indices // window_rows if window_rows else np.zeros_like(indices)
                )
                for group in np.unique(groups):
                    slots = np.flatnonzero(groups == group)
                    replacement = self._read_rows(indices[slots], split)
                    for key in values:
                        values[key][rows[slots]] = replacement[key]
        return values

    def get_batch(self, batch_size: int, split: str = "train") -> dict[str, np.ndarray]:
        """Read sequentially; job callers use the node-clocked async stream."""
        if not hasattr(self, "_streams"):
            self._streams: dict[tuple[int, str], Iterator[dict[str, np.ndarray]]] = {}
        key = (batch_size, split)
        if key not in self._streams:
            reader = copy.copy(self)
            reader._streams = {}
            if hasattr(reader, "_reader"):
                reader._reader = copy.copy(reader._reader)
            if hasattr(reader, "_buffers"):
                reader._buffers = {}
            self._streams[key] = batches(
                [reader], [1.0], batch_size, split, validate=False
            )
        return next(self._streams[key])
