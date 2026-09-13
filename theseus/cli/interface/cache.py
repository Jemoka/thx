"""Local read-through cache for immutable object-store value parts."""

import os
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from threading import Lock
from uuid import uuid4

from theseus.store import ObjectReader


_COPY_WORKERS = 32


class ObjectCache:
    """Mirror finalized Parquet parts locally and expose a read-only store."""

    def __init__(self, root: Path, cache_dir: Path | None = None) -> None:
        self.root = root.expanduser().absolute()
        self.source = self.root / "objects" / "values"
        if cache_dir is None:
            cache_home = Path(
                os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
            ).expanduser()
            root_key = sha256(os.fsencode(str(self.root))).hexdigest()
            cache_dir = cache_home / "theseus" / "ui" / root_key / "values"
        self.path = cache_dir
        self.path.mkdir(parents=True, exist_ok=True)
        self.reader = ObjectReader(self.root / "objects", self.path)
        self.version = 0
        self._parts = self._part_names(self.path)
        self._poll_lock = Lock()

    def poll(self) -> int:
        """Synchronize finalized parts and return the resulting cache version."""
        with self._poll_lock:
            before = self._part_names(self.path)
            source = self._part_names(self.source)
            missing = source - before
            if missing:
                with ThreadPoolExecutor(
                    max_workers=min(_COPY_WORKERS, len(missing))
                ) as executor:
                    tuple(executor.map(self._copy_part, missing))

            for name in before - source:
                (self.path / name).unlink(missing_ok=True)

            parts = self._part_names(self.path)
            if parts != self._parts:
                self._parts = parts
                self.version += 1
            return self.version

    @staticmethod
    def _part_names(path: Path) -> frozenset[str]:
        try:
            entries = os.scandir(path)
        except FileNotFoundError:
            return frozenset()
        with entries:
            return frozenset(
                entry.name
                for entry in entries
                if not entry.name.startswith(".") and entry.name.endswith(".parquet")
            )

    def _copy_part(self, name: str) -> None:
        pending = self.path / f".{name}.{uuid4().hex}.tmp"
        try:
            copyfile(self.source / name, pending)
            pending.replace(self.path / name)
        except FileNotFoundError:
            # Compaction may retire a part after the source directory was listed.
            pass
        finally:
            pending.unlink(missing_ok=True)
