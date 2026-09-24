# metagross/_span.py
"""Public op-span API: mark named regions the tracer attaches to GPU events."""
from __future__ import annotations

import contextlib
import threading
import time

from metagross import _profile


@contextlib.contextmanager
def span(name: str):
    emit = _profile.current_emitter()
    if emit is None:
        yield  # not running under metagross: no-op, keep the script runnable
        return
    tid = threading.get_native_id()
    emit.span_begin(tid, time.monotonic_ns(), str(name))
    try:
        yield
    finally:
        emit.span_end(tid, time.monotonic_ns())
