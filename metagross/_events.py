# metagross/_events.py
"""Event enrichment, attribution, and rendering for Metagross."""
from __future__ import annotations

import bisect
import dataclasses
import datetime
import heapq
import itertools
import json as _json
import os
import shlex
import time
import unicodedata

from metagross import _bpf, _viewer, MetagrossError


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
    """Per-TID profile logs with incremental point-in-time stack replay.

    A thread is kept only while its log holds a record, so threads that
    have finished cost nothing.
    """

    def __init__(self):
        self._logs: dict[int, list[tuple]] = {}
        self._states: dict[int, _ReplayState] = {}
        self._horizon = 0  # no query below this instant can be answered
        # Queries refused because the history for that instant is gone. The
        # thread may well have been inside a project function.
        self.refused = 0

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
        if ts_ns < self._horizon:
            self.refused += 1
            return None  # history below the prune horizon is gone; do not guess
        log = self._logs.get(tid)
        if not log:
            return None
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
        self._horizon = max(self._horizon, ts_ns)
        for tid, log in self._logs.items():
            log[:] = [record for record in log if record[0] >= ts_ns]
            self._states[tid] = _ReplayState()
        self._forget_empty()

    def _forget_empty(self) -> None:
        for tid in [tid for tid, log in self._logs.items() if not log]:
            del self._logs[tid]
            self._states.pop(tid, None)

    def prune(self, min_ts_ns: int) -> None:
        # The horizon only ever moves forward: history a prior gap or prune
        # already put out of reach must not become trusted again just
        # because a later prune call happens to pass a lower min_ts_ns
        # (e.g. Joiner.flush pruning to the oldest still-held event after
        # on_gap raised the horizon on a dropped record).
        self._horizon = max(self._horizon, min_ts_ns)
        for tid, log in self._logs.items():
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
        self._forget_empty()


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
        self._horizon = 0

    def on_span(self, tid: int, ts_ns: int, name: str | None) -> None:
        log = self._logs.setdefault(tid, [])
        log.insert(bisect.bisect_right(log, ts_ns, key=_report_ts),
                   (ts_ns, name))

    def attribute(self, tid: int, ts_ns: int):
        log = self._logs.get(tid)
        if not log or ts_ns < self._horizon:
            return None  # history below the prune horizon is gone; do not guess
        index = bisect.bisect_right(log, ts_ns, key=_report_ts)
        return log[index - 1][1] if index else None

    def on_gap(self, ts_ns: int) -> None:
        # The lost record may have been the report that ended a span, so no
        # thread's earlier state can be trusted until its next report.
        self._horizon = max(self._horizon, ts_ns)
        for log in self._logs.values():
            log[:] = [report for report in log if report[0] >= ts_ns]
        self._forget_empty()

    def _forget_empty(self) -> None:
        for tid in [tid for tid, log in self._logs.items() if not log]:
            del self._logs[tid]

    def prune(self, min_ts_ns: int) -> None:
        self._horizon = max(self._horizon, min_ts_ns)
        for log in self._logs.values():
            cut = bisect.bisect_left(log, min_ts_ns, key=_report_ts)
            if cut and log[cut - 1][1] is not None:
                cut -= 1  # still the active span at the horizon
            del log[:cut]
        self._forget_empty()


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
    def __init__(self, hold_ns: int = 100_000_000,
                 max_wait_ns: int = 5_000_000_000,
                 max_pending: int = 200_000):
        self.hold_ns = hold_ns
        # Unread profile data is capped, so a busy tracer is a few seconds
        # behind at most. A call that waits longer than this, or behind this
        # many others, is given up on; the target must not be able to make
        # the tracer hold calls without limit.
        self.max_wait_ns = max_wait_ns
        self.max_pending = max_pending
        self.timeline = FrameTimeline()
        self.spans = OpSpanTimeline()
        self.registry = KernelRegistry()
        self.allocs = AllocTracker()
        # A heap ordered by entry time, so a flush touches only the calls it
        # releases however many are waiting.
        self._pending: list = []
        self._arrivals = itertools.count()  # orders calls with equal times
        self._newest_profile_ns = 0  # newest timestamp in the profile stream
        self._waited_out = 0

    @property
    def refused_attributions(self) -> int:
        """Calls left unknown because the tracer lacked their history."""
        return self.timeline.refused + self._waited_out

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
        heapq.heappush(
            self._pending, (raw.ts, next(self._arrivals), raw, api, kernel))

    def on_profile_record(self, rec) -> None:
        tag = rec[0]
        if tag == "frame":
            _, kind, tid, ts, func, path, line = rec
            self.timeline.on_record(kind, tid, ts, func, path, line)
        elif tag == "span":
            _, tid, ts, name = rec
            self.spans.on_span(tid, ts, name)
        elif tag == "gap":
            # A thread reads the clock before it takes the write lock, so the
            # record that reveals a hole can be older than one written before
            # the hole. Nothing already seen may survive the gap.
            ts = max(rec[1] if rec[1] is not None else time.monotonic_ns(),
                     self._newest_profile_ns + 1)
            self.timeline.on_gap(ts)
            self.spans.on_gap(ts)
        else:
            return
        self._newest_profile_ns = max(self._newest_profile_ns, ts)

    def enrich(self, event: AttributedEvent) -> EnrichedEvent:
        kernel, details = describe(
            event.api, event.raw, self.registry, self.allocs,
            kernel_override=event.kernel_at_enqueue,
        )
        return EnrichedEvent(
            event.raw, event.api, event.frame, kernel, details, event.span
        )

    def flush(self, now_ns: int, force: bool = False,
              delivered_until_ns: int | None = None,
              profile_drained_ns: int | None = None):
        """Release events whose calling stack is known and prune old history.

        `delivered_until_ns` is an instant by which every call that had
        returned has been handed to `on_gpu_event`: the time just before the
        ring buffer was last drained. `profile_drained_ns` is an instant at
        which the profile stream was empty, so every record written before
        it has been handed to `on_profile_record`. Both default to `now_ns`.
        """
        if delivered_until_ns is None:
            delivered_until_ns = now_ns
        if profile_drained_ns is None:
            profile_drained_ns = now_ns
        # A thread writes its profile records before it enters the driver, so
        # its stack is known once the stream has been read past that entry: a
        # newer record has arrived, or the stream was empty after it. Until
        # then a record may still be on its way, or missing with the hole not
        # yet revealed, and the stack seen so far would be a guess.
        profile_until_ns = max(self._newest_profile_ns, profile_drained_ns)
        ready_until_ns = min(profile_until_ns, now_ns - self.hold_ns)
        pending = self._pending
        out = []
        while pending and (force or pending[0][0] <= ready_until_ns):
            _, _, raw, api, kernel = heapq.heappop(pending)
            # The calling thread stays inside the driver for the whole call,
            # so its stack at return is its stack at entry. Query at return:
            # a long call's entry can be older than the pruned history, but
            # its return is always recent.
            returned_ns = raw.ts + raw.dur
            out.append(AttributedEvent(
                raw, api, self.timeline.attribute(raw.tid, returned_ns),
                kernel, self.spans.attribute(raw.tid, returned_ns)))
        while pending and (now_ns - pending[0][0] >= self.max_wait_ns
                           or len(pending) > self.max_pending):
            # The profile stream is too far behind to wait for.
            _, _, raw, api, kernel = heapq.heappop(pending)
            self._waited_out += 1
            out.append(AttributedEvent(raw, api, None, kernel))
        # History must reach back to every call still to be attributed: the
        # ones held here, and the ones not yet delivered, which returned
        # after the ring buffer was last drained. The controller may have
        # spent longer than a hold window since then, so the clock alone is
        # not enough. Prune even when nothing was released, or an idle GPU
        # lets the log grow.
        horizon = min(now_ns, delivered_until_ns)
        if pending:
            horizon = min(horizon, pending[0][0])
        self.timeline.prune(horizon - self.hold_ns)
        self.spans.prune(horizon - self.hold_ns)
        return out


def _hex(v: int) -> str:
    return f"0x{v:x}"


def describe(api, ev, registry: KernelRegistry, allocs: AllocTracker,
             kernel_override=_UNSET):
    """Describe a raw CUDA event.

    For a launch, kernel_override is the name the registry held when the
    call was queued (what Joiner.enrich passes), or None to keep the
    placeholder rather than trust a later registration. Left unset, the
    name is looked up in registry now.
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
                 lost_profile_records: int = 0,
                 refused_attributions: int = 0,
                 libcuda_mismatch: bool = False) -> dict:
        complete = not any((lost_events, dropped_nested_calls,
                            render_failed, trace_failed,
                            lost_profile_records, refused_attributions,
                            libcuda_mismatch))
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
                "refused_attributions": refused_attributions,
                "libcuda_mismatch": libcuda_mismatch,
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


# The API column fits the longest name, `StreamSynchronize`.
_COLUMNS = (("TIME", 12), ("FUNCTION", 18), ("LOCATION", 20),
            ("API", 18), ("RET", 5), ("DURATION", 9))


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
                 str(ev.raw.ret), _viewer._duration(ev.raw.dur))
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
