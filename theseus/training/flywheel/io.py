"""Schedule dataset I/O at the level where each loader knows what it needs.

Opening or mapping a file gives access to its bytes; scattered requests can
still incur repeated remote waits. These helpers let callers describe multiple
reads together so independent I/O can overlap. They serve two different access
patterns, using the same process-wide pool of eight workers:

* PMD uses ``read_ranges(source, destination, ranges)``. It already owns a
  contiguous token-window cache and knows which 4 MiB blocks are missing.
  Supply disjoint, contiguous array slices: each copies ``source[span]`` into
  ``destination[span]``. Slice offsets are array elements, not byte offsets.
  The call waits for all copies; unrequested destination regions are untouched.
  PMD marks those blocks loaded afterward and gathers samples from its cached
  strided view. Fetching individual rows through RowReader would bypass this
  block-reuse strategy. The caller owns allocation, cache validity, and eviction;
  read_ranges only fills the requested regions and keeps no cache of its own.

* PaddedDataset and ContrastivePaddedDataset use ``RowReader``. Their sampler
  requests arbitrary rows from paired token/mask files, without a PMD window
  cache. Map the files once, then pass the mappings and each batch's row IDs to
  ``read``. It returns gathered arrays in the requested order, choosing between
  direct mapped indexing and sorted, coalesced positional reads based on
  observed cost. Its class docstring gives a complete usage example.

Use read_ranges when you already manage a buffer and its missing regions;
use RowReader when you need a batch of rows assembled from mapped files. Neither
helper chooses samples or changes their logical order. Files must remain
immutable; ordinary sequential file reads are sufficient for a full-file scan.
"""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import os
from typing import BinaryIO

import numpy as np
from numpy.typing import DTypeLike

# Shared across train/validation and mixtures: never create a pool per reader.
_READERS = ThreadPoolExecutor(max_workers=8, thread_name_prefix="flywheel-io")


def read_ranges(
    source: np.ndarray, destination: np.ndarray, ranges: list[slice]
) -> None:
    """Fill disjoint contiguous ranges of a caller-owned buffer."""
    pending = [
        _READERS.submit(np.copyto, destination[span], source[span]) for span in ranges
    ]
    # Drain all reads even on failure, so no worker outlives the destination.
    error = None
    for future in pending:
        try:
            future.result()
        except Exception as exc:
            error = exc
    if error is not None:
        raise error


class RowReader:
    """Fetch batches of scattered rows from large, immutable token/mask files.

    Think of each file as a table and each training batch as a list of row IDs.
    Opening the file gives you a byte stream; assembling a random batch yourself
    means translating those IDs into offsets and issuing reads. On a remote
    filesystem, scattered reads can spend far more time waiting than copying
    bytes. Memory mapping makes indexing convenient, but uncached rows still
    require filesystem reads.

    This reader sees the whole batch request, so it can sort physical accesses,
    combine nearby reads, and overlap independent requests across token and mask
    files. It restores the requested row order afterward, including duplicates.
    It starts with ordinary mapped indexing and tries parallel I/O after slow
    reads, keeping it only when observed timings justify the overhead. For a
    sequential file scan, ordinary file reads are sufficient; use this for
    repeated random batches whose I/O benefits from being scheduled together.

    Reuse one instance per sequential loader worker. Call
    ``map(path, dtype, shape)`` once for each raw, row-major file, using its
    on-disk dtype and two-dimensional shape. Then call
    ``samples, padding = reader.read((tokens, masks), indices)`` for each batch.
    Sources may also be contiguous column slices of those mappings. Mapping
    does not eagerly load the whole file; the OS manages cached file pages.

    Example for paired token and mask files, each with 100,000 rows and 1,025
    columns. These are the stored dimensions, including any shifted-label
    token. Create the reader and mappings outside the batch loop::

        reader = RowReader()
        shape = (
            100_000,
            1025,
        )
        tokens = reader.map(
            "train.bin",
            np.uint32,
            shape,
        )
        masks = reader.map(
            "train.bin.mask",
            np.bool_,
            shape,
        )
        sources = (
            tokens,
            masks,
        )

        # Supply row IDs from your sampler for each batch.
        indices = np.array(
            [9, 2, 9],
            dtype=np.int64,
        )
        samples, padding = (
            reader.read(
                sources,
                indices,
            )
        )

    Both outputs have shape (3, 1025), in row order 9, 2, 9. For subsequent
    batches, call ``reader.read(sources, next_indices)`` with the same reader
    and mappings; do not reopen the files or construct a reader per batch.

    The caller supplies valid, nonnegative row indices and owns sampling/RNG
    state. Returned arrays are writable copies, in source-tuple and index order,
    with the original dtypes. Keep files unchanged and the reader alive across
    batches to reuse open files and timing measurements. Do not share its
    scheduling state between concurrent consumers. Scheduling can change how
    bytes are fetched, but never which samples are returned.
    """

    def __init__(self) -> None:
        self._parallel = False
        self._serial_cost = 0.0
        self._parallel_cost = 0.0
        self._reads = 0
        self._files: dict[str, BinaryIO] = {}

    def map(self, path: str, dtype: DTypeLike, shape: tuple[int, ...]) -> np.memmap:
        """Keep the mapping and parallel reads on the same open file."""
        filename = os.path.abspath(path)
        if filename not in self._files:
            self._files[filename] = open(filename, "rb", buffering=0)
        return np.memmap(self._files[filename], dtype=dtype, mode="r", shape=shape)

    def read(
        self, sources: tuple[np.ndarray, ...], indices: np.ndarray
    ) -> list[np.ndarray]:
        start = perf_counter()
        if not self._parallel or len(indices) == 0:
            results = [source[indices] for source in sources]
        else:
            order = np.argsort(indices, kind="stable")
            results = [
                np.empty((len(indices), *source.shape[1:]), dtype=source.dtype)
                for source in sources
            ]
            pending = []
            for source, result in zip(sources, results):
                # pread releases the GIL even for one row, independently of
                # NumPy's advanced-index gather implementation.
                root = source
                while isinstance(root.base, np.memmap):
                    root = root.base
                assert isinstance(root, np.memmap)
                filename = str(root.filename)
                if filename not in self._files:
                    self._files[filename] = open(filename, "rb", buffering=0)
                offset = root.offset + source.ctypes.data - root.ctypes.data
                for slots in np.array_split(order, min(8, len(order))):
                    pending.append(
                        _READERS.submit(
                            _read_rows,
                            self._files[filename],
                            offset,
                            source.strides[0],
                            indices[slots],
                            result,
                            slots,
                        )
                    )
            error = None
            for future in pending:
                try:
                    future.result()
                except Exception as exc:
                    error = exc
            if error is not None:
                raise error
        elapsed = perf_counter() - start
        cost = elapsed / max(1, len(indices))
        if self._parallel:
            self._parallel_cost = cost
        else:
            self._serial_cost = cost
        self._reads += 1
        # Probe parallel I/O after a slow read, retaining it only if it beats
        # the observed serial cost. Recheck periodically as cache state changes.
        self._parallel = elapsed > 0.01 and (
            not self._parallel_cost
            or self._parallel_cost < self._serial_cost
            or self._reads % 32 == 0
        )
        return results


def _read_rows(
    file: BinaryIO,
    offset: int,
    stride: int,
    indices: np.ndarray,
    result: np.ndarray,
    slots: np.ndarray,
) -> None:
    """Coalesce nearby sorted rows, bounding gaps and each read's footprint."""
    width = result.shape[1] * result.dtype.itemsize
    cursor = 0
    while cursor < len(indices):
        first = offset + int(indices[cursor]) * stride
        stop = first + width
        end = cursor + 1
        while end < len(indices):
            following = offset + int(indices[end]) * stride
            if following > stop + 65536 or following + width - first > 4 * 1024 * 1024:
                break
            stop = following + width
            end += 1
        chunks = []
        position = first
        while position < stop:
            chunk = os.pread(file.fileno(), stop - position, position)
            if not chunk:
                raise EOFError(f"Dataset ended during a read at byte {position}")
            chunks.append(chunk)
            position += len(chunk)
        raw = chunks[0] if len(chunks) == 1 else b"".join(chunks)
        rows = np.ndarray(
            ((stop - first - width) // stride + 1, result.shape[1]),
            dtype=result.dtype,
            buffer=raw,
            strides=(stride, result.dtype.itemsize),
        )
        result[slots[cursor:end]] = rows[indices[cursor:end] - indices[cursor]]
        cursor = end
