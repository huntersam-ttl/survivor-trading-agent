"""Single-cycle overlap protection.

A lock file (~/.tradingagents/survivor/cycle.lock) is created exclusively
(O_EXCL). If it already exists and is fresh, the next trigger skips with
CYCLE_ALREADY_RUNNING. Stale locks (older than max_age_sec) are broken.
"""

from __future__ import annotations

import os
import time

LOCK_DIR = os.path.join(os.path.expanduser("~"), ".tradingagents", "survivor")
LOCK_PATH = os.path.join(LOCK_DIR, "cycle.lock")


class CycleLock:
    def __init__(self, lock_path: str | None = None, stale_age_sec: int = 3600):
        self.lock_path = lock_path or LOCK_PATH
        self.stale_age_sec = stale_age_sec
        self._held = False

    def acquire(self) -> bool:
        """Try to acquire the cycle lock non-blockingly. True on success."""
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        try:
            if os.path.exists(self.lock_path):
                age = time.time() - os.path.getmtime(self.lock_path)
                if age <= self.stale_age_sec:
                    return False  # another cycle is (still) running
                os.remove(self.lock_path)  # break stale lock
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self._held = True
            return True
        except FileExistsError:
            return False

    def release(self) -> None:
        if self._held and os.path.exists(self.lock_path):
            os.remove(self.lock_path)
        self._held = False

    def __enter__(self) -> CycleLock:
        return self

    def __exit__(self, *exc) -> None:
        self.release()
