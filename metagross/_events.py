# metagross/_events.py
"""Event enrichment, attribution, and rendering for Metagross."""
from __future__ import annotations

import bisect
import dataclasses
import datetime
import json as _json
import os
import shlex
import time
import unicodedata

from metagross import _bpf, MetagrossError


_UNSET = object()


class KernelRegistry:
    def __init__(self):
        self._names: dict[int, str] = {}

    def observe(self, api, ev) -> None:
        if ev.ret != 0 or not ev.out:
            return
        if api.base == "cuKernelGetFunction":
            known = self._names.get(ev.args[1])
            if known:
                self._names[ev.out] = known
        else:
            name = ev.name.decode("utf-8", "replace")
            if name:
                self._names[ev.out] = name

    def name(self, handle: int) -> str | None:
        return self._names.get(handle)


class AllocTracker:
    def __init__(self):
        self._sizes: dict[int, int] = {}
        self.total_bytes = 0

    def on_alloc(self, ptr: int, size: int) -> None:
        if ptr:
            self._sizes[ptr] = size
            self.total_bytes += size

    def on_free(self, ptr: int) -> int | None:
        size = self._sizes.pop(ptr, None)
        if size is not None:
            self.total_bytes -= size
        return size


@dataclasses.dataclass(frozen=True)
class FrameInfo:
    function: str
    file: str
    line: int


class _ReplayState:
    __slots__ = ("index", "last_ts_ns", "stack")

    def __init__(self):
        self.index = 0
        self.last_ts_ns = -1
        self.stack: list[FrameInfo] = []


class FrameTimeline:
    """Per-TID profile logs with incremental point-in-time stack replay."""

    def __init__(self):
        self._logs: dict[int, list[tuple]] = {}
        self._states: dict[int, _ReplayState] = {}
        self._horizons: dict[int, int] = {}

    @staticmethod
    def _apply(stack: list[FrameInfo], record: tuple) -> None:
        _, kind, func, path, line = record
        if kind == 0:
            stack.append(FrameInfo(func, path, line))
        elif stack and stack[-1] == FrameInfo(func, path, line):
            stack.pop()
        # A return that does not match the top is spurious or reordered;
        # leave the stack unchanged rather than unwinding a live frame.

    def on_record(self, kind, tid, ts_ns, func, path, line) -> None:
        record = (ts_ns, kind, func, path, line)
        log = self._logs.setdefault(tid, [])
        if log and record < log[-1]:
            bisect.insort_right(log, record)
        else:
            log.append(record)
        state = self._states.get(tid)
        if state is not None and ts_ns <= state.last_ts_ns:
            # A late record invalidates the incremental stack. The next
            # monotonic query rebuilds it once from the now-sorted log.
            self._states[tid] = _ReplayState()

    def attribute(self, tid: int, ts_ns: int):
        log = self._logs.get(tid)
        if not log:
            return None
        horizon = self._horizons.get(tid)
        if horizon is not None and ts_ns < horizon:
            return None  # history below the prune horizon is gone; do not guess
        state = self._states.setdefault(tid, _ReplayState())
        if ts_ns < state.last_ts_ns:
            # GPU delivery can rarely exceed the hold window. Answer an older
            # query independently rather than rewinding the monotonic cursor.
            stack: list[FrameInfo] = []
            end = bisect.bisect_right(log, (ts_ns, 2))
            for record in log[:end]:
                self._apply(stack, record)
            return stack[-1] if stack else None

        while state.index < len(log) and log[state.index][0] <= ts_ns:
            self._apply(state.stack, log[state.index])
            state.index += 1
        state.last_ts_ns = ts_ns
        return state.stack[-1] if state.stack else None

    def on_gap(self, ts_ns: int) -> None:
        # Drop ALL pre-gap records, including still-open CALLs -- not just
        # closed history the way prune() does. A CALL whose matching RETURN
        # was the very record the gap dropped looks indistinguishable from a
        # genuinely still-open CALL: prune-and-retain would keep it, and
        # because `_apply` only pops on an exact top-of-stack match, that
        # leaked frame would never clear and every top-level-idle query
        # at/after the gap would confidently (and wrongly) attribute to it
        # forever. Dropping everything below the gap forces post-gap queries
        # to <unknown> until a real CALL re-establishes the stack.
        for tid, log in self._logs.items():
            log[:] = [record for record in log if record[0] >= ts_ns]
            self._horizons[tid] = max(self._horizons.get(tid, ts_ns), ts_ns)
            self._states[tid] = _ReplayState()

    def prune(self, min_ts_ns: int) -> None:
        for tid, log in self._logs.items():
            # A horizon only ever moves forward: history a prior gap or
            # prune already put out of reach must not become trusted again
            # just because a later prune call happens to pass a lower
            # min_ts_ns (e.g. Joiner.flush pruning to the oldest still-held
            # event after on_gap raised the horizon on a dropped record).
            self._horizons[tid] = max(self._horizons.get(tid, min_ts_ns), min_ts_ns)
            cut = 0
            open_stack: list[FrameInfo] = []
            open_indices: list[int] = []
            for i, record in enumerate(log):
                if record[0] >= min_ts_ns:
                    break
                cut = i + 1
                before = len(open_stack)
                self._apply(open_stack, record)
                if len(open_stack) > before:
                    open_indices.append(i)           # a CALL was pushed
                elif len(open_stack) < before:
                    open_indices.pop()               # a RETURN popped it
            if not cut:
                continue
            retained = [log[i] for i in open_indices]  # still-open calls, in order
            log[:] = retained + log[cut:]
            state = self._states.get(tid)
            if state is not None:
                if state.index >= cut:
                    state.index = len(retained) + (state.index - cut)
                else:
                    self._states[tid] = _ReplayState()


def _report_ts(report: tuple) -> int:
    return report[0]


class OpSpanTimeline:
    """Each thread's active `metagross.span()` name over time.

    The target reports a thread's active span whenever it changes, so the
    span at a point in time is the last report at or before it. Like
    `FrameTimeline`, a lookup below the horizon fails closed.
    """

    def __init__(self):
        self._logs: dict[int, list[tuple[int, str | None]]] = {}
        self._horizons: dict[int, int] = {}

    def on_span(self, tid: int, ts_ns: int, name: str | None) -> None:
        log = self._logs.setdefault(tid, [])
        log.insert(bisect.bisect_right(log, ts_ns, key=_report_ts),
                   (ts_ns, name))

    def attribute(self, tid: int, ts_ns: int):
        log = self._logs.get(tid)
        if not log:
            return None
        horizon = self._horizons.get(tid)
        if horizon is not None and ts_ns < horizon:
            return None  # history below the prune horizon is gone; do not guess
        index = bisect.bisect_right(log, ts_ns, key=_report_ts)
        return log[index - 1][1] if index else None

    def on_gap(self, ts_ns: int) -> None:
        # The lost record may have been the report that ended a span, so no
        # thread's earlier state can be trusted until its next report.
        for tid, log in self._logs.items():
            log[:] = [report for report in log if report[0] >= ts_ns]
            self._horizons[tid] = max(self._horizons.get(tid, ts_ns), ts_ns)

    def prune(self, min_ts_ns: int) -> None:
        for tid, log in self._logs.items():
            self._horizons[tid] = max(self._horizons.get(tid, min_ts_ns), min_ts_ns)
            cut = bisect.bisect_left(log, min_ts_ns, key=_report_ts)
            if cut and log[cut - 1][1] is not None:
                cut -= 1  # still the active span at the horizon
            del log[:cut]


@dataclasses.dataclass
class AttributedEvent:
    raw: object
    api: object
    frame: FrameInfo | None
    kernel_at_enqueue: str | None = None
    span: str | None = None


@dataclasses.dataclass
class EnrichedEvent:
    raw: object
    api: object
    frame: FrameInfo | None
    kernel: str | None
    details: dict
    span: str | None = None


class Joiner:
    def __init__(self, hold_ns: int = 100_000_000):
        self.hold_ns = hold_ns
        self.timeline = FrameTimeline()
        self.spans = OpSpanTimeline()
        self.registry = KernelRegistry()
        self.allocs = AllocTracker()
        self._pending: list = []

    def on_gpu_event(self, raw) -> None:
        api = _bpf.API_BY_ID.get(raw.api_id)
        if api is None:
            return
        if api.category == "register":
            self.registry.observe(api, raw)
            return
        kernel = None
        if api.category in ("launch", "launch_ex"):
            handle = raw.args[0] if api.category == "launch" else raw.args[1]
            kernel = self.registry.name(handle)
        self._pending.append((raw.ts, raw, api, kernel))

    def on_profile_record(self, rec) -> None:
        tag = rec[0]
        if tag == "frame":
            _, kind, tid, ts, func, path, line = rec
            self.timeline.on_record(kind, tid, ts, func, path, line)
        elif tag == "span":
            _, tid, ts, name = rec
            self.spans.on_span(tid, ts, name)
        elif tag == "gap":
            gap_ts = rec[1] if rec[1] is not None else time.monotonic_ns()
            self.timeline.on_gap(gap_ts)
            self.spans.on_gap(gap_ts)

    def enrich(self, event: AttributedEvent) -> EnrichedEvent:
        kernel, details = describe(
            event.api, event.raw, self.registry, self.allocs,
            kernel_override=event.kernel_at_enqueue,
        )
        return EnrichedEvent(
            event.raw, event.api, event.frame, kernel, details, event.span
        )

    def flush(self, now_ns: int, force: bool = False,
              delivered_until_ns: int | None = None):
        """Release events held for a full window and prune old history.

        `delivered_until_ns` is an instant by which every call that had
        returned has been handed to `on_gpu_event`: the time just before the
        ring buffer was last drained. It defaults to `now_ns`.
        """
        released, kept = [], []
        for ts, raw, api, kernel in self._pending:
            if force or now_ns - ts >= self.hold_ns:
                released.append((ts, raw, api, kernel))
            else:
                kept.append((ts, raw, api, kernel))
        self._pending = kept
        released.sort(key=lambda item: item[0])
        out = []
        for ts, raw, api, kernel in released:
            # The calling thread stays inside the driver for the whole call,
            # so its stack at return is its stack at entry. Query at return:
            # a long call's entry can be older than the pruned history, but
            # its return is always recent.
            returned_ns = ts + raw.dur
            out.append(AttributedEvent(
                raw, api, self.timeline.attribute(raw.tid, returned_ns),
                kernel, self.spans.attribute(raw.tid, returned_ns)))
        # Held events returned within the last hold window, and events not
        # yet delivered returned after the ring buffer was last drained. The
        # controller may have spent longer than a hold window since then, so
        # history is kept back to that drain, not to now. Prune even when
        # nothing was released, or an idle GPU lets the log grow.
        if delivered_until_ns is None:
            delivered_until_ns = now_ns
        horizon = min(now_ns, delivered_until_ns) - self.hold_ns
        self.timeline.prune(horizon)
        self.spans.prune(horizon)
        return out


def _hex(v: int) -> str:
    return f"0x{v:x}"


def describe(api, ev, registry: KernelRegistry, allocs: AllocTracker,
             kernel_override=_UNSET):
    """Describe a raw CUDA event.

    kernel_override distinguishes two callers for launch categories:
    left at the default _UNSET (the direct-call path DescribeTest uses),
    the kernel name is looked up live in registry, matching pre-existing
    behavior. Passed explicitly (the Joiner.enrich path, always passed),
    a str names the kernel snapshotted at enqueue time and None freezes
    the launch to the placeholder rather than trusting a later
    registration.
    """
    cat = api.category
    kernel = None
    det: dict = {}
    if cat in ("launch", "launch_ex"):
        if cat == "launch":
            handle = ev.args[0]
            gx, gy, gz = (ev.args[1] & 0xFFFFFFFF, ev.args[2] & 0xFFFFFFFF,
                         ev.args[3] & 0xFFFFFFFF)
            bx, by, bz = (ev.args[4] & 0xFFFFFFFF, ev.args[5] & 0xFFFFFFFF,
                         ev.args[6] & 0xFFFFFFFF)
            det["grid"] = f"{gx},{gy},{gz}"
            det["block"] = f"{bx},{by},{bz}"
            det["shared"] = ev.args[7] & 0xFFFFFFFF
            det["stream"] = _hex(ev.args[8])
        else:
            handle = ev.args[1]
            gx, gy = ev.args[2] & 0xFFFFFFFF, ev.args[2] >> 32
            gz, bx = ev.args[3] & 0xFFFFFFFF, ev.args[3] >> 32
            by, bz = ev.args[4] & 0xFFFFFFFF, ev.args[4] >> 32
            det["grid"] = f"{gx},{gy},{gz}"
            det["block"] = f"{bx},{by},{bz}"
            det["shared"] = ev.args[5]
            det["stream"] = _hex(ev.args[6])
        if kernel_override is _UNSET:
            kernel = registry.name(handle) or f"kernel@{handle:#x}"
        else:
            kernel = kernel_override or f"kernel@{handle:#x}"
        det["function_handle"] = _hex(handle)
    elif cat == "graph_launch":
        # One call replays every kernel captured in the graph, so there is
        # no single kernel to name.
        det["graph_exec"] = _hex(ev.args[0])
        det["stream"] = _hex(ev.args[1])
    elif cat in ("alloc", "alloc_async"):
        det["bytes"] = ev.args[1]
        det["ptr"] = _hex(ev.out)
        if cat == "alloc_async":
            det["stream"] = _hex(ev.args[2])
        if ev.ret == 0:
            allocs.on_alloc(ev.out, ev.args[1])
        det["gpu_total"] = allocs.total_bytes
    elif cat in ("free", "free_async"):
        det["ptr"] = _hex(ev.args[0])
        if ev.ret == 0:
            size = allocs.on_free(ev.args[0])
            if size is not None:
                det["bytes"] = size
        if cat == "free_async":
            det["stream"] = _hex(ev.args[1])
        det["gpu_total"] = allocs.total_bytes
    elif cat.startswith("copy"):
        det["bytes"] = ev.args[2]
        if api.base.endswith("Async"):
            det["stream"] = _hex(ev.args[3])
    elif cat == "sync":
        if api.base == "cuStreamSynchronize":
            det["stream"] = _hex(ev.args[0])
        elif api.base == "cuEventSynchronize":
            det["event"] = _hex(ev.args[0])
    return kernel, det


@dataclasses.dataclass
class _Aggregate:
    count: int = 0
    errors: int = 0
    total_duration_ns: int = 0
    max_duration_ns: int = 0
    successful_bytes: int = 0

    def observe(self, event: EnrichedEvent) -> None:
        self.count += 1
        self.errors += event.raw.ret != 0
        self.total_duration_ns += event.raw.dur
        self.max_duration_ns = max(self.max_duration_ns, event.raw.dur)
        if event.raw.ret == 0:
            self.successful_bytes += int(event.details.get("bytes", 0))

    def fields(self) -> dict:
        return {
            "count": self.count,
            "errors": self.errors,
            "total_duration_ns": self.total_duration_ns,
            "max_duration_ns": self.max_duration_ns,
            "successful_bytes": self.successful_bytes,
        }


class CaptureStats:
    """Aggregate enriched events without changing event rendering."""

    def __init__(self):
        self.events = 0
        self.attributed = 0
        self.cuda_errors = 0
        self.total_api_duration_ns = 0
        self.synchronization_duration_ns = 0
        self.successful_allocation_bytes = 0
        self.observed_peak_bytes = 0
        self._api: dict[str, _Aggregate] = {}
        self._function: dict[tuple[str, str, int], _Aggregate] = {}
        self._kernel: dict[str, _Aggregate] = {}
        self._span: dict[str, _Aggregate] = {}
        self._copy_bytes: dict[str, int] = {}

    def observe(self, event: EnrichedEvent) -> None:
        self.events += 1
        self.attributed += event.frame is not None
        self.cuda_errors += event.raw.ret != 0
        self.total_api_duration_ns += event.raw.dur
        if event.api.category == "sync":
            self.synchronization_duration_ns += event.raw.dur
        if event.api.category in ("alloc", "alloc_async") and event.raw.ret == 0:
            self.successful_allocation_bytes += int(event.details.get("bytes", 0))
        gpu_total = event.details.get("gpu_total")
        if isinstance(gpu_total, int):
            self.observed_peak_bytes = max(self.observed_peak_bytes, gpu_total)
        if event.api.category.startswith("copy") and event.raw.ret == 0:
            copied = int(event.details.get("bytes", 0))
            self._copy_bytes[event.api.base] = (
                self._copy_bytes.get(event.api.base, 0) + copied
            )

        self._api.setdefault(event.api.base, _Aggregate()).observe(event)
        if event.frame is not None:
            function_key = (
                event.frame.function, event.frame.file, event.frame.line
            )
            self._function.setdefault(function_key, _Aggregate()).observe(event)
        if event.kernel is not None and not event.kernel.startswith("kernel@"):
            self._kernel.setdefault(event.kernel, _Aggregate()).observe(event)
        if event.span is not None:
            self._span.setdefault(event.span, _Aggregate()).observe(event)

    @staticmethod
    def _top_rows(groups: dict, field_names: tuple[str, ...]) -> list[dict]:
        ordered = sorted(
            groups.items(),
            key=lambda item: (
                -item[1].count, -item[1].total_duration_ns, item[0]
            ),
        )
        rows = []
        for key, aggregate in ordered[:20]:
            values = key if isinstance(key, tuple) else (key,)
            row = dict(zip(field_names, values))
            row.update(aggregate.fields())
            rows.append(row)
        return rows

    def snapshot(self, *, lost_events: int, dropped_nested_calls: int,
                 observed_outstanding_bytes: int, render_failed: bool,
                 trace_failed: bool = False,
                 lost_profile_records: int = 0) -> dict:
        complete = not any((lost_events, dropped_nested_calls,
                            render_failed, trace_failed,
                            lost_profile_records))
        return {
            "schema_version": 1,
            "complete": complete,
            "capture": {
                "events": self.events,
                "attributed": self.attributed,
                "unknown_attribution": self.events - self.attributed,
                "cuda_errors": self.cuda_errors,
                "lost_events": lost_events,
                "dropped_nested_calls": dropped_nested_calls,
                "render_failed": render_failed,
                "trace_failed": trace_failed,
                "lost_profile_records": lost_profile_records,
            },
            "timing": {
                "total_api_duration_ns": self.total_api_duration_ns,
                "synchronization_duration_ns": self.synchronization_duration_ns,
            },
            "memory": {
                "successful_allocation_bytes": self.successful_allocation_bytes,
                "observed_peak_bytes": self.observed_peak_bytes,
                "observed_outstanding_bytes": observed_outstanding_bytes,
            },
            "copies": {
                "successful_bytes_by_api": dict(sorted(self._copy_bytes.items())),
            },
            "apis": [
                {"api": api, **aggregate.fields()}
                for api, aggregate in sorted(self._api.items())
            ],
            "top_functions": self._top_rows(
                self._function, ("function", "file", "line")
            ),
            "top_kernels": self._top_rows(self._kernel, ("kernel",)),
            "top_spans": self._top_rows(self._span, ("span",)),
        }


def shell_quote_details(details: dict) -> str:
    parts = []
    for key, value in details.items():
        text = value if isinstance(value, str) else str(value)
        parts.append(f"{key}={shlex.quote(text)}")
    return " ".join(parts)


def printable(text: str) -> str:
    """Replace control characters; names and paths come from the target."""
    return "".join(
        "?" if unicodedata.category(character).startswith("C") else character
        for character in text)


_COLUMNS = (("TIME", 12), ("FUNCTION", 18), ("LOCATION", 20),
            ("API", 16), ("RET", 5), ("DURATION", 9))


def event_record(
    event: EnrichedEvent, wall_minus_mono_ns: int, pid: int
) -> dict:
    """Return the stable JSON event record shared by all machine sinks."""
    kernel = event.kernel
    if kernel is not None and kernel.startswith("kernel@"):
        kernel = None
    wall_ns = event.raw.ts + wall_minus_mono_ns
    moment = datetime.datetime.fromtimestamp(wall_ns / 1e9).astimezone()
    return {
        "timestamp": moment.isoformat(),
        "pid": pid,
        "tid": event.raw.tid,
        "function": event.frame.function if event.frame else None,
        "file": event.frame.file if event.frame else None,
        "line": event.frame.line if event.frame else None,
        "api": event.api.base,
        "kernel": kernel,
        "return_code": event.raw.ret,
        "duration_ns": event.raw.dur,
        "details": event.details,
        "span": event.span,
    }


_TABLE_KERNEL_CHARS = 127  # a table row stays readable; JSONL has the full name


class Renderer:
    def __init__(self, stream, json_output, wall_minus_mono_ns, pid):
        self.stream = stream
        self.json_output = json_output
        self.wall_minus_mono_ns = wall_minus_mono_ns
        self.pid = pid

    def header(self) -> None:
        if not self.json_output:
            self._write("".join(n.ljust(w) for n, w in _COLUMNS) + "DETAILS\n")
            self.flush()

    def emit(self, ev: EnrichedEvent) -> None:
        if self.json_output:
            record = event_record(ev, self.wall_minus_mono_ns, self.pid)
            self._write(_json.dumps(record, separators=(",", ":")) + "\n")
            return
        kernel, details = ev.kernel, ev.details
        wall_ns = ev.raw.ts + self.wall_minus_mono_ns
        moment = datetime.datetime.fromtimestamp(wall_ns / 1e9).astimezone()
        if kernel is not None:
            if len(kernel) > _TABLE_KERNEL_CHARS:
                kernel = kernel[:_TABLE_KERNEL_CHARS - 3] + "..."
            details = {"kernel": kernel, **details}
            details.pop("function_handle", None)
        if ev.span is not None:
            details = {**details, "span": ev.span}
        func = ev.frame.function if ev.frame else "<unknown>"
        loc = (f"{os.path.basename(ev.frame.file)}:{ev.frame.line}"
               if ev.frame else "<unknown>")
        cells = (moment.strftime("%H:%M:%S.") + f"{moment.microsecond // 10000:02d}",
                 func, loc, ev.api.base.removeprefix("cu"),
                 str(ev.raw.ret), f"{ev.raw.dur / 1e6:.2f}ms")
        row = "".join(c.ljust(w) if len(c) < w else c + " "
                      for c, (_, w) in zip(cells, _COLUMNS))
        self._write(printable(row + shell_quote_details(details)) + "\n")

    def flush(self) -> None:
        try:
            self.stream.flush()
        except OSError as exc:
            raise MetagrossError(f"trace output failed: {exc}") from None

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
        except OSError as exc:
            raise MetagrossError(f"trace output failed: {exc}") from None
