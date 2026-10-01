# metagross/_span.py
"""Public op-span API: mark named regions the tracer attaches to GPU events."""
from __future__ import annotations

import contextlib

from metagross import _profile


@contextlib.contextmanager
def span(name: str):
    if _profile.current_emitter() is None:
        yield  # not running under metagross: no-op, keep the script runnable
        return
    entry = _profile.OpenSpan(str(name))
    outer = _profile.open_spans.get()
    mine = outer + (entry,)
    _profile.open_spans.set(mine)
    _profile.report_span()
    try:
        yield
    finally:
        # Tasks and threads that inherited this span stop reporting it.
        entry.name = None
        current = _profile.open_spans.get()
        if current is mine:
            _profile.open_spans.set(outer)
        elif entry in current:
            # Closed while another span was innermost: none of the spans
            # open in this context can be trusted any more.
            for other in current:
                other.name = None
        _profile.report_span()
