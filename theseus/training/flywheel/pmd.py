"""
pmd.py
"Poor Man's Dataloader" Datasets
"""

import os
from pathlib import Path
from typing import Optional

import jax
import numpy as np

from theseus.base.job import ExecutionSpec
from theseus.training.flywheel.dataset import Dataset
from theseus.training.flywheel.io import read_ranges

BYTES_PER_BLOCK = 4 * 1024 * 1024  # 4MB block size
BUFFER_BLOCKS = 64  # Number of blocks to buffer (256MB total, ~64M tokens)


class MemmapDataset(Dataset):
    def __init__(
        self, spec: ExecutionSpec, block_size: int, name: str, suffix: str = ""
    ):
        self.cache_train: Optional[np.memmap] = None
        self.cache_val: Optional[np.memmap] = None
        self.block_size = block_size
        self.process_index = jax.process_index()
        self.process_count = max(1, jax.process_count())

        data_dir: Path = spec.hardware.hosts[self.process_index].cluster.data_dir
        path: Path = data_dir / name if suffix == "" else data_dir / f"{name}_{suffix}"
        self.path = path

        self.has_val = (path / "val.bin").exists()
        self._buffers: dict[str, tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _get_memmap(self, split: str) -> np.memmap:
        """Get or create memmap for the given split."""
        if split == "train":
            if self.cache_train is None:
                self.cache_train = np.memmap(
                    os.path.join(self.path, "train.bin"), dtype=np.uint32, mode="r"
                )
            result = self.cache_train
        else:
            if self.cache_val is None:
                self.cache_val = np.memmap(
                    os.path.join(self.path, "val.bin"), dtype=np.uint32, mode="r"
                )
            result = self.cache_val
        assert result is not None
        return result

    def _size(self, split: str) -> int:
        if split == "val" and not self.has_val:
            split = "train"
        size = (len(self._get_memmap(split)) - 1) // self.block_size
        if size < 1:
            raise ValueError(f"Dataset too small for block_size={self.block_size}")
        return size

    @property
    def _window_rows(self) -> int:
        return max(1, BYTES_PER_BLOCK * BUFFER_BLOCKS // (4 * self.block_size))

    def _plan(self, split: str, rng: np.random.Generator, count: int) -> np.ndarray:
        """Shuffle samples in a size-weighted, contiguous window of at most 256 MiB."""
        size = self._size(split)
        rows = self._window_rows
        start = int(rng.integers(size)) // rows * rows
        return start + np.resize(rng.permutation(min(rows, size - start)), count)

    def _read_rows(self, ix: np.ndarray, split: str) -> dict[str, np.ndarray]:
        if split == "val" and not self.has_val:
            split = "train"
        rows = self._window_rows
        start = int(ix[0]) // rows * rows
        cached = self._buffers.get(split)
        data = self._get_memmap(split)
        source = data[start * self.block_size : (start + rows) * self.block_size + 1]
        # Keep the same sample window, but fetch only the storage blocks touched
        # by this batch. Short validation runs and resumed hosts need not read
        # all 256 MiB before producing their first sample.
        tokens_per_read = max(1, BYTES_PER_BLOCK // data.dtype.itemsize)
        if cached is None or cached[0] != start:
            buffer = np.empty(source.shape, dtype=source.dtype)
            loaded = np.zeros(
                (len(source) + tokens_per_read - 1) // tokens_per_read,
                dtype=np.bool_,
            )
            windows = np.lib.stride_tricks.sliding_window_view(
                buffer, self.block_size + 1
            )[:: self.block_size]
            self._buffers[split] = (start, buffer, loaded, windows)
        else:
            _, buffer, loaded, windows = cached
        if not loaded.all():
            offsets = (ix - start) * self.block_size
            # Usually a sample touches one or two blocks. Vectorize this hot path
            # without allocating a token-sized index matrix.
            first = offsets // tokens_per_read
            span = (self.block_size + tokens_per_read - 1) // tokens_per_read
            needed = np.unique(
                np.minimum(
                    first[:, None] + np.arange(span + 1),
                    ((offsets + self.block_size) // tokens_per_read)[:, None],
                )
            )
            missing = needed[~loaded[needed]]
            read_ranges(
                source,
                buffer,
                [
                    slice(
                        int(block) * tokens_per_read, (int(block) + 1) * tokens_per_read
                    )
                    for block in missing
                ],
            )
            loaded[missing] = True
        samples = windows[ix - start]
        x = samples[:, :-1].astype(np.int64)
        y = samples[:, 1:].astype(np.int64)
        return {"x": x, "y": y, "padding_mask": np.ones_like(x, dtype=np.bool_)}
