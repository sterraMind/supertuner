"""Run cancellation primitives.

Two independent cancellation channels exist because neither is reliable on its
own:

- a signal (SIGINT/SIGTERM/SIGBREAK) delivered to the engine process. `os.kill`
  with SIGTERM becomes `TerminateProcess` on Windows, so the handler is a best
  effort there.
- a flag file inside the run directory (`<run_dir>/cancel`, `cancel.flag` is
  accepted too). The server creates it, which works identically on every OS and
  is also how a human cancels a run started from a terminal.

The training loop checks `ctx.cancel.check()` between micro-batches, and the
subprocess runner polls `ctx.cancel.poll()` while a child step is executing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

#: Canonical flag file name inside a run directory.
CANCEL_FLAG_NAMES = ("cancel", "cancel.flag")


class Cancelled(RuntimeError):
    """Raised inside a step when cancellation was requested."""


@dataclass
class CancelToken:
    """Cooperative cancellation state shared by the runtime and the backends."""

    run_dir: Path | None = None
    reason: str = ""
    _requested: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def requested(self) -> bool:
        return self._requested

    def request(self, reason: str = "cancelled by request") -> None:
        with self._lock:
            if not self._requested:
                self._requested = True
                self.reason = reason

    def flag_paths(self) -> list[Path]:
        if self.run_dir is None:
            return []
        return [self.run_dir / name for name in CANCEL_FLAG_NAMES]

    def poll(self) -> bool:
        """Check the in-process flag and the flag files; returns True once cancelled."""
        if not self._requested and self.run_dir is not None:
            for path in self.flag_paths():
                try:
                    if path.exists():
                        self.request(f"cancel flag file present: {path.name}")
                        break
                except OSError:
                    continue
        return self._requested

    def check(self) -> None:
        """Raise `Cancelled` when a cancellation is pending."""
        if self.poll():
            raise Cancelled(self.reason or "cancelled")