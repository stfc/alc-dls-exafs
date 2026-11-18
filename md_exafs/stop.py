"""Cooperative stop request for runs that a scheduler may kill.

Slurm sends SIGTERM when a job reaches its time limit and SIGKILL shortly
after.  The handler installed by :func:`install_sigterm_handler` only sets
:data:`STOP`.  Chunk loops check it at chunk boundaries, finish the chunk in
flight, and exit with :class:`RunStoppedError`.  Library code never installs the
handler; command-line entry points do.
"""

from __future__ import annotations

import signal
import threading

#: Set once a stop has been requested.  Nothing clears it except callers (tests).
STOP = threading.Event()

_STOP_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGUSR1") if hasattr(signal, name)
)


class RunStoppedError(RuntimeError):
    """A run stopped on request before all of its work was done."""


def _request_stop(signum: int, frame: object) -> None:
    STOP.set()


def install_sigterm_handler() -> bool:
    """Make SIGTERM and SIGUSR1 request a stop.  Safe to call repeatedly.

    Returns False, and changes nothing, when called off the main thread, where
    Python does not allow installing signal handlers.
    """
    try:
        for sig in _STOP_SIGNALS:
            signal.signal(sig, _request_stop)
    except ValueError:
        return False
    return True
