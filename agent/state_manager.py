"""
Unified state management — single coordination point for all JSON state files.

Provides:
- FileStore: generic JSON file read/write with mtime caching, atomic writes, and locking
- Global cache invalidation
- Consolidated migration logic
"""

import asyncio
import copy
import json
import logging
import os
import threading
import weakref
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", dict, list)


class FileStore:
    """Generic JSON file store with mtime-based caching, atomic writes, and locking.

    Each instance manages a single JSON file on disk. Reads are cached in memory
    and only re-parsed when the file's mtime changes. Writes use atomic
    tmp-file + os.replace to avoid corruption on crash.

    Thread safety is provided by a per-instance threading.Lock. All FileStore
    instances are tracked in a class-level list so ``invalidate_all()`` can
    clear every cache in one call.
    """

    _instances: weakref.WeakSet["FileStore"] = weakref.WeakSet()

    def __init__(self, path: Path, default_factory: Callable[[], Any] = dict) -> None:
        self._path = Path(path)
        self._default_factory = default_factory
        self._lock = threading.Lock()
        self._cached: Any | None = None
        self._cache_mtime: float = 0.0
        FileStore._instances.add(self)

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> Any:
        """Return the file contents (deep-copied), or the default if missing/corrupt.

        Uses mtime-based caching: the file is only re-read from disk when its
        modification time has changed since the last read.
        """
        with self._lock:
            if not self._path.exists():
                return self._default_factory()
            try:
                mtime = os.stat(self._path).st_mtime
                if self._cached is not None and mtime == self._cache_mtime:
                    return copy.deepcopy(self._cached)
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._cached = data
                self._cache_mtime = mtime
                return copy.deepcopy(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read %s: %s", self._path.name, e)
                return self._default_factory()

    def write(self, data: Any) -> None:
        """Atomically write *data* to the file and update the in-memory cache.

        Writes to a temporary file first, then uses ``os.replace`` to swap it
        into place. This ensures that a crash mid-write never leaves a
        half-written file on disk.
        """
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(
                f".tmp_{os.getpid()}_{threading.get_ident()}"
            )
            tmp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(str(tmp_path), str(self._path))
            self._cached = copy.deepcopy(data)
            self._cache_mtime = os.stat(self._path).st_mtime

    def invalidate(self) -> None:
        """Clear the in-memory cache so the next ``read()`` re-reads from disk."""
        with self._lock:
            self._cached = None
            self._cache_mtime = 0.0

    @classmethod
    def invalidate_all(cls) -> None:
        """Clear the caches of every live FileStore instance.

        Dead references are automatically pruned by WeakSet, so iterating
        only visits instances that are still alive.
        """
        for store in list(cls._instances):
            store.invalidate()


async def async_invalidate_all() -> None:
    """Async-friendly wrapper around ``FileStore.invalidate_all()``."""
    await asyncio.to_thread(FileStore.invalidate_all)
