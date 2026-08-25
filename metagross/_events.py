# metagross/_events.py
"""Event enrichment, attribution, and rendering for Metagross."""
from __future__ import annotations

import bisect
import dataclasses
import datetime
import json as _json
import os
import shlex

from metagross import _bpf, MetagrossError


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


class FrameTimeline:
    """Per-TID log of profile records, replayed to answer point-in-time queries."""

    def __init__(self):
        self._logs: dict[int, list[tuple]] = {}

    def on_record(self, kind, tid, ts_ns, func, path, line) -> None:
        self._logs.setdefault(tid, []).append((ts_ns, kind, func, path, line))

    def attribute(self, tid: int, ts_ns: int):
        log = self._logs.get(tid)
        if not log:
            return None
        stack: list[FrameInfo] = []
        idx = bisect.bisect_right(log, (ts_ns, 2))
        for _, kind, func, path, line in log[:idx]:
            if kind == 0:
                stack.append(FrameInfo(func, path, line))
            elif stack and stack[-1].function == func:
                stack.pop()
            elif stack:
                stack.pop()  # unwind mismatch conservatively
        return stack[-1] if stack else None

    def prune(self, min_ts_ns: int) -> None:
        # ponytail: O(n) replay per attribute + periodic prune; index it if
        # profiles of long-running loops ever measure slow.
        for tid, log in self._logs.items():
            depth = 0
            cut = 0
            for i, (ts, kind, *_rest) in enumerate(log):
                if ts >= min_ts_ns:
                    break
                depth += 1 if kind == 0 else -1
                if depth <= 0:
                    depth = max(depth, 0)
                    cut = i + 1
            if cut:
                self._logs[tid] = log[cut:]


@dataclasses.dataclass
class AttributedEvent:
    raw: object
    api: object
    frame: FrameInfo | None


class Joiner:
    def __init__(self, hold_ns: int = 100_000_000):
        self.hold_ns = hold_ns
        self.timeline = FrameTimeline()
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
        self._pending.append((raw.ts, raw, api))

    def on_profile_record(self, rec) -> None:
        self.timeline.on_record(*rec)

    def flush(self, now_ns: int, force: bool = False):
        released, kept = [], []
        for ts, raw, api in self._pending:
            if force or now_ns - ts >= self.hold_ns:
                released.append((ts, raw, api))
            else:
                kept.append((ts, raw, api))
        self._pending = kept
        released.sort(key=lambda item: item[0])
        out = [AttributedEvent(raw, api, self.timeline.attribute(raw.tid, ts))
               for ts, raw, api in released]
        if released:
            self.timeline.prune(min(ts for ts, _, _ in kept) if kept else now_ns)
        return out


def _hex(v: int) -> str:
    return f"0x{v:x}"


def describe(api, ev, registry: KernelRegistry, allocs: AllocTracker):
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
        kernel = registry.name(handle) or f"kernel@{handle:#x}"
        det["function_handle"] = _hex(handle)
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


def shell_quote_details(details: dict) -> str:
    parts = []
    for key, value in details.items():
        text = value if isinstance(value, str) else str(value)
        parts.append(f"{key}={shlex.quote(text)}")
    return " ".join(parts)


_COLUMNS = (("TIME", 12), ("FUNCTION", 18), ("LOCATION", 20),
            ("API", 16), ("RET", 5), ("DURATION", 9))


class Renderer:
    def __init__(self, stream, json_output, wall_minus_mono_ns, pid, joiner):
        self.stream = stream
        self.json_output = json_output
        self.wall_minus_mono_ns = wall_minus_mono_ns
        self.pid = pid
        self.joiner = joiner

    def header(self) -> None:
        if not self.json_output:
            self._write("".join(n.ljust(w) for n, w in _COLUMNS) + "DETAILS\n")

    def emit(self, ev) -> None:
        kernel, details = describe(ev.api, ev.raw, self.joiner.registry,
                                   self.joiner.allocs)
        wall_ns = ev.raw.ts + self.wall_minus_mono_ns
        moment = datetime.datetime.fromtimestamp(
            wall_ns / 1e9).astimezone()
        if self.json_output:
            if kernel is not None and kernel.startswith("kernel@"):
                kernel = None
            record = {
                "timestamp": moment.isoformat(),
                "pid": self.pid,
                "tid": ev.raw.tid,
                "function": ev.frame.function if ev.frame else None,
                "file": ev.frame.file if ev.frame else None,
                "line": ev.frame.line if ev.frame else None,
                "api": ev.api.base,
                "kernel": kernel,
                "return_code": ev.raw.ret,
                "duration_ns": ev.raw.dur,
                "details": details,
            }
            self._write(_json.dumps(record, separators=(",", ":")) + "\n")
            return
        if kernel is not None:
            details = {"kernel": kernel, **details}
            details.pop("function_handle", None)
        func = ev.frame.function if ev.frame else "<unknown>"
        loc = (f"{os.path.basename(ev.frame.file)}:{ev.frame.line}"
               if ev.frame else "<unknown>")
        cells = (moment.strftime("%H:%M:%S.") + f"{moment.microsecond // 10000:02d}",
                 func, loc, ev.api.base.removeprefix("cu"),
                 str(ev.raw.ret), f"{ev.raw.dur / 1e6:.2f}ms")
        row = "".join(c.ljust(w) if len(c) < w else c + " "
                      for c, (_, w) in zip(cells, _COLUMNS))
        self._write(row + shell_quote_details(details) + "\n")

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except OSError as exc:
            raise MetagrossError(f"trace output failed: {exc}") from None
