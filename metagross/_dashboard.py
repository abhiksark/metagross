# metagross/_dashboard.py
"""Private runner for the dashboard that `metagross --web` starts.

The controller execs this module as the unprivileged target user in a new
session. It reads the producer token from an inherited pipe, serves the
in-memory receiver on 127.0.0.1, and reports the bound port on a second pipe.
"""

from __future__ import annotations

import ctypes
import os
import signal

from metagross import _web

_PR_SET_PDEATHSIG = 1
_MAX_TOKEN_BYTES = 128
_RECENT_EVENTS = 500
_REFRESH_SECONDS = 0.2


def main(argv: list[str]) -> int:
    token_fd, ready_fd, port, controller_pid = (int(value) for value in argv)
    # Die with the controller. This must follow the credential drop, which
    # clears the parent-death signal, so it lives here rather than before exec.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
        return 1
    if os.getppid() != controller_pid:
        return 1  # the controller died before the signal was armed

    with os.fdopen(token_fd, "rb") as token_file:
        token = token_file.read(_MAX_TOKEN_BYTES + 1).decode("ascii")

    def report_ready(bound_port: int) -> None:
        with os.fdopen(ready_fd, "w") as ready:
            ready.write(f"{bound_port}\n")

    return _web.run_web_dashboard(
        None,
        None,
        _RECENT_EVENTS,
        _REFRESH_SECONDS,
        port,
        ingest_token=token,
        on_ready=report_ready,
    )
