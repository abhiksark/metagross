# metagross/_profile.py
"""Profiling hooks run inside the traced child; record codec (wire v3)."""
from __future__ import annotations

import contextvars
import os
import queue
import struct
import sys
import threading
import time

# Frame "kind" values used by FrameTimeline.on_record and everywhere a
# decoded frame record is consumed. They are not the wire rtype (see below),
# just call-vs-return direction.
CALL, RETURN = 0, 1

# Wire record types (the `rtype` byte in the common header). CALL and RETURN
# here are deliberately not reused as names: `_FRAME_CALL_RTYPE` /
# `_FRAME_RETURN_RTYPE` carry the CALL/RETURN distinction on the wire, kept
# private because nothing outside this module needs the raw byte value; the
# public `CALL`/`RETURN` kind constants above must stay 0/1 for
# FrameTimeline.on_record and the many tests that call it directly.
HELLO = 0
FRAME_DEF = 1
_FRAME_CALL_RTYPE = 2
_FRAME_RETURN_RTYPE = 3
SPAN_SET = 4    # a thread's active span is now this name
SPAN_CLEAR = 5  # a thread has no active span
HOOK_REPLACED = 6  # the target swapped out the profiling hook

_WIRE_VERSION = 3
_SEQ_MOD = 2**32
_MAX_STR = 500
_MAX_FRAMES = 1 << 16  # distinct frames the controller keeps per capture
_MAX_QUEUED_CHUNKS = 256  # 64 KiB reads the controller may fall behind by

# Every record starts with this 6-byte common header.
_COMMON = struct.Struct("<BIB")  # rtype, seq, _reserved
_HELLO_BODY = struct.Struct("<BIQ")  # version, pid, start_ns
_FRAME_DEF_PREFIX = struct.Struct("<IIIHH")  # frame_id, line, _pad, func_len, path_len
# CALL/RETURN carry only the interned frame_id; FRAME_DEF (above) is what
# assigns func/path/line to that id, once, the first time a frame is seen.
_FRAME_REF_BODY = struct.Struct("<IQI")  # tid, ts_ns, frame_id
_SPAN_SET_PREFIX = struct.Struct("<IQH")  # tid, ts_ns, name_len
_SPAN_CLEAR_BODY = struct.Struct("<IQ")  # tid, ts_ns
_HOOK_REPLACED_BODY = struct.Struct("<Q")  # ts_ns

_FRAME_RTYPE_BY_KIND = {CALL: _FRAME_CALL_RTYPE, RETURN: _FRAME_RETURN_RTYPE}
_KIND_BY_FRAME_RTYPE = {_FRAME_CALL_RTYPE: CALL, _FRAME_RETURN_RTYPE: RETURN}

_seq = 0


def _next_seq() -> int:
    """Return the next per-child sequence number and advance the counter.

    Not internally locked: callers that can race (the profiling hook, which
    threading.setprofile fires on every target thread) must hold their own
    lock around `_next_seq()` and the matching write so a record's seq and
    its position in the pipe stay in the same order. Single-threaded callers
    (encoders, tests) need nothing extra.
    """
    global _seq
    seq = _seq
    _seq = (_seq + 1) % _SEQ_MOD
    return seq


def _reset_seq() -> None:
    """Reset the module-level child seq counter. Test-only."""
    global _seq
    _seq = 0


def record_type(record_bytes: bytes) -> int:
    """Return the rtype byte of an encoded record.

    Exposed for later tasks (e.g. asserting which record a raw blob is)
    without every caller re-deriving the header layout.
    """
    return _COMMON.unpack_from(record_bytes)[0]


def encode_hello(pid: int, start_ns: int) -> bytes:
    seq = _next_seq()
    return (_COMMON.pack(HELLO, seq, 0)
            + _HELLO_BODY.pack(_WIRE_VERSION, pid & 0xFFFFFFFF,
                               start_ns & 0xFFFFFFFFFFFFFFFF))


def _encode_frame_def(frame_id, func, path, line) -> bytes:
    func_bytes = func.encode("utf-8", "replace")[:_MAX_STR]
    path_bytes = path.encode("utf-8", "replace")[:_MAX_STR]
    seq = _next_seq()
    return (_COMMON.pack(FRAME_DEF, seq, 0)
            + _FRAME_DEF_PREFIX.pack(frame_id, line, 0, len(func_bytes), len(path_bytes))
            + func_bytes + path_bytes)


def _encode_frame_ref(kind, tid, ts_ns, frame_id) -> bytes:
    rtype = _FRAME_RTYPE_BY_KIND[kind]
    seq = _next_seq()
    return _COMMON.pack(rtype, seq, 0) + _FRAME_REF_BODY.pack(tid, ts_ns, frame_id)


def encode_frame(kind, tid, ts_ns, func, path, line) -> bytes:
    """Encode one self-contained frame record: a fresh FRAME_DEF (id 0)
    immediately followed by the CALL/RETURN that references it.

    A test or tool convenience for exercising the wire codec without
    running a whole `_FrameEmitter`. `install()`'s hot path does not call
    this -- it interns real frames through `_FrameEmitter` instead, so a
    repeated frame costs one small reference rather than a redefinition.
    """
    frame_id = 0
    return (_encode_frame_def(frame_id, func, path, line)
            + _encode_frame_ref(kind, tid, ts_ns, frame_id))


class _FrameEmitter:
    """Intern (func, path, line) frames to a small integer id.

    Writes one FRAME_DEF the first time a frame is seen, then a CALL/RETURN
    carrying only that frame_id thereafter. `write` is called once per
    finished record's bytes (never given a merged blob), so a caller that
    wants to inspect or drop-and-count individual records can do so.

    Owns its own lock, held across every `_next_seq()` + `write()` pair
    (`emit` and `span` alike). Both run on every target thread, so without
    one lock they race on the module-global `_next_seq()` counter and the
    interleaving of their writes, producing out-of-order or duplicate seqs
    on the wire and manufacturing false gaps. No reentrancy hazard: `emit`
    only runs inside the profiling hook, where Python does not re-enter the
    hook, and `report_span` records a thread's new span before it calls
    `span`, so the hook has nothing left to report from inside the write.
    """

    def __init__(self, write):
        self._write = write
        self._frames: dict[tuple[str, str, int], int] = {}
        self._next_id = 0
        self._lock = threading.Lock()

    def emit(self, kind, tid, ts, func, path, line) -> None:
        key = (func, path, line)
        with self._lock:
            frame_id = self._frames.get(key)
            if frame_id is None:
                frame_id = self._next_id
                self._next_id += 1
                self._frames[key] = frame_id
                self._write(_encode_frame_def(frame_id, func, path, line))
            self._write(_encode_frame_ref(kind, tid, ts, frame_id))

    def span(self, tid, ts, name) -> None:
        with self._lock:
            self._write(encode_span(tid, ts, name))

    def hook_replaced(self, ts) -> None:
        with self._lock:
            self._write(encode_hook_replaced(ts))


class _DropCountWriter:
    """Write profile records through a non-blocking fd; drop-and-count
    ordinary records under pipe overrun, but never a FRAME_DEF.

    A FRAME_DEF assigns the frame_id later CALL/RETURN records reference;
    dropping one would desync the reader's intern map forever (a CALL for
    an id the reader never saw defined), so it is retried until it fits.
    That can briefly block the target -- accepted, since definitions are
    rare and small and the alternative (a permanently wrong frame_id) is
    far worse. Every other record type is best-effort: on `BlockingIOError`
    (EAGAIN, pipe full) it is dropped and counted rather than blocking the
    target's own thread on tracer backpressure.
    """

    def __init__(self, fd: int, os_write=os.write):
        self._fd = fd
        self._os_write = os_write
        self.dropped = 0

    def write(self, data: bytes) -> None:
        # Every record is well under PIPE_BUF (the largest, a FRAME_DEF,
        # is at most ~6 + 16 + 2*_MAX_STR bytes), so a successful os.write
        # on a pipe is always atomic here: the return value is never a
        # partial write that would itself desync the reader's byte stream.
        if record_type(data) in (FRAME_DEF, HOOK_REPLACED):
            # A lost HOOK_REPLACED would leave stale frames trusted.
            self._write_never_dropping(data)
            return
        try:
            self._os_write(self._fd, data)
        except BlockingIOError:
            self.dropped += 1
        except OSError:
            pass  # broken trace pipe must never kill the target

    def _write_never_dropping(self, data: bytes) -> None:
        while True:
            try:
                self._os_write(self._fd, data)
                return
            except BlockingIOError:
                # Pipe momentarily full: retry until it drains. This runs
                # under `_FrameEmitter`'s lock, so yield the GIL between
                # attempts rather than busy-spinning and starving other
                # target threads' emits for however long the pipe stays full.
                time.sleep(0)
                continue
            except OSError:
                return  # broken trace pipe must never kill the target


def encode_span(tid, ts_ns, name) -> bytes:
    """Encode the span now active on a thread; `None` means no span."""
    seq = _next_seq()
    if name is None:
        return (_COMMON.pack(SPAN_CLEAR, seq, 0)
                + _SPAN_CLEAR_BODY.pack(tid, ts_ns))
    name_bytes = name.encode("utf-8", "replace")[:_MAX_STR]
    return (_COMMON.pack(SPAN_SET, seq, 0)
            + _SPAN_SET_PREFIX.pack(tid, ts_ns, len(name_bytes)) + name_bytes)


def encode_hook_replaced(ts_ns) -> bytes:
    seq = _next_seq()
    return _COMMON.pack(HOOK_REPLACED, seq, 0) + _HOOK_REPLACED_BODY.pack(ts_ns)


class RecordReader:
    """Decode the v2 profile wire format into tagged tuples.

    Every record carries a per-child `seq`. A `seq` that does not match what
    this reader expects next means one or more records never arrived (a
    full pipe, a dropped write, or a lost HELLO); on a mismatch it emits a
    synthetic `("gap", ts_ns)` tuple BEFORE the record that revealed the
    gap, so a caller (the Joiner) can fail closed before trusting anything
    past the hole. `_expected` starts as None (no HELLO seen yet), so if the
    very first record is not a HELLO, that alone is treated as a gap rather
    than silently trusting an unannounced stream.
    """

    def __init__(self):
        self._buf = b""
        self._expected = None
        self.version = None
        # frame_id -> (func, path, line). Never cleared, including on a
        # gap: FRAME_DEF is never dropped (_DropCountWriter makes it
        # undroppable), so a definition always precedes its use and ids
        # never desync.
        self._frames: dict[int, tuple] = {}
        # Total records lost to gaps: the sum of seq deltas across every
        # detected gap, plus the (should-not-happen) defensive drops below.
        # This is the authoritative loss count surfaced in the summary --
        # the writer's own `_DropCountWriter.dropped` exists only for
        # testing the writer in isolation, because the child cannot
        # reliably flush a final count once its pipe is overrunning.
        self.lost_records = 0
        self.hook_replacements = 0
        # A gap revealed on a record with no ts_ns of its own (FRAME_DEF,
        # HELLO) is not reported immediately: it is held here and emitted
        # with the ts of the next record that DOES carry a real ts_ns, so
        # the Joiner's fail-closed horizon is as tight as possible instead
        # of falling back to "now" at decode time. `finalize()` flushes it
        # as `("gap", None)` if the stream ends before that happens.
        self._pending_gap = False

    def feed(self, data: bytes) -> list[tuple]:
        self._buf += data
        out: list[tuple] = []
        while len(self._buf) >= _COMMON.size:
            rtype, seq, _reserved = _COMMON.unpack_from(self._buf)
            if rtype == HELLO:
                need = _COMMON.size + _HELLO_BODY.size
                if len(self._buf) < need:
                    break
                version, _pid, _start_ns = _HELLO_BODY.unpack_from(
                    self._buf, _COMMON.size)
                self.version = version
                if self._expected is None:
                    # The very first record ever: this HELLO establishes
                    # the baseline itself, not a gap against one.
                    self._expected = (seq + 1) % _SEQ_MOD
                else:
                    # A HELLO seen after the stream already started --
                    # never happens in production (the child sends exactly
                    # one), but route it through the same gap check as
                    # every other rtype for a uniform invariant.
                    self._note_seq(seq, out, None)
                self._buf = self._buf[need:]
                continue
            if rtype == FRAME_DEF:
                prefix_off = _COMMON.size
                if len(self._buf) < prefix_off + _FRAME_DEF_PREFIX.size:
                    break
                frame_id, line, _pad, fl, pl = _FRAME_DEF_PREFIX.unpack_from(
                    self._buf, prefix_off)
                if fl > _MAX_STR or pl > _MAX_STR:
                    return self._corrupt(out)  # the hook never writes this
                body_off = prefix_off + _FRAME_DEF_PREFIX.size
                total = body_off + fl + pl
                if len(self._buf) < total:
                    break
                func = self._buf[body_off:body_off + fl].decode("utf-8", "replace")
                path = self._buf[body_off + fl:total].decode("utf-8", "replace")
                self._buf = self._buf[total:]
                # FRAME_DEF carries no ts_ns of its own; a gap revealed here
                # has an unknown ts until the next timestamped record.
                self._note_seq(seq, out, None)
                if frame_id in self._frames or len(self._frames) < _MAX_FRAMES:
                    self._frames[frame_id] = (func, path, line)
                continue
            if rtype in (_FRAME_CALL_RTYPE, _FRAME_RETURN_RTYPE):
                need = _COMMON.size + _FRAME_REF_BODY.size
                if len(self._buf) < need:
                    break
                tid, ts_ns, frame_id = _FRAME_REF_BODY.unpack_from(
                    self._buf, _COMMON.size)
                self._buf = self._buf[need:]
                self._note_seq(seq, out, ts_ns)
                frame = self._frames.get(frame_id)
                if frame is None:
                    # A frame past the _MAX_FRAMES cap, or a corrupt id.
                    # Fail closed -- drop this one record and count it,
                    # never guess a frame, and never clear the map on the
                    # strength of one bad id.
                    self.lost_records += 1
                    continue
                kind = _KIND_BY_FRAME_RTYPE[rtype]
                out.append(("frame", kind, tid, ts_ns, *frame))
                continue
            if rtype == SPAN_SET:
                prefix_off = _COMMON.size
                if len(self._buf) < prefix_off + _SPAN_SET_PREFIX.size:
                    break
                tid, ts_ns, nl = _SPAN_SET_PREFIX.unpack_from(
                    self._buf, prefix_off)
                if nl > _MAX_STR:
                    return self._corrupt(out)
                body_off = prefix_off + _SPAN_SET_PREFIX.size
                total = body_off + nl
                if len(self._buf) < total:
                    break
                name = self._buf[body_off:total].decode("utf-8", "replace")
                self._buf = self._buf[total:]
                self._note_seq(seq, out, ts_ns)
                out.append(("span", tid, ts_ns, name))
                continue
            if rtype == SPAN_CLEAR:
                need = _COMMON.size + _SPAN_CLEAR_BODY.size
                if len(self._buf) < need:
                    break
                tid, ts_ns = _SPAN_CLEAR_BODY.unpack_from(self._buf, _COMMON.size)
                self._buf = self._buf[need:]
                self._note_seq(seq, out, ts_ns)
                out.append(("span", tid, ts_ns, None))
                continue
            if rtype == HOOK_REPLACED:
                need = _COMMON.size + _HOOK_REPLACED_BODY.size
                if len(self._buf) < need:
                    break
                (ts_ns,) = _HOOK_REPLACED_BODY.unpack_from(self._buf, _COMMON.size)
                self._buf = self._buf[need:]
                self._note_seq(seq, out, ts_ns)
                # Frames open on that thread will never be seen to return;
                # treat it like lost records and forget what came before.
                self.hook_replacements += 1
                self.lost_records += 1
                out.append(("gap", ts_ns))
                continue
            return self._corrupt(out)  # unknown rtype
        return out

    def _corrupt(self, out: list) -> list:
        """Drop a stream that cannot be resynchronized, and fail closed.

        An unknown rtype or an impossible length leaves no boundary to trust;
        never raise, wait for more bytes, or spin on the same ones.
        """
        self._buf = b""
        out.append(("gap", None))
        return out

    def _note_seq(self, seq: int, out: list, ts_ns) -> None:
        mismatch = self._expected is None or seq != self._expected
        if mismatch:
            if self._expected is not None:
                self.lost_records += (seq - self._expected) % _SEQ_MOD
            self._pending_gap = True
        if self._pending_gap and ts_ns is not None:
            out.append(("gap", ts_ns))
            self._pending_gap = False
        self._expected = (seq + 1) % _SEQ_MOD

    def finalize(self) -> list:
        """Flush an unresolved pending gap once the stream ends.

        A gap first revealed on a no-ts record is normally reported with
        the ts of the next real-ts record (see `_note_seq`). If the
        stream ends (EOF) before one arrives, that tighter horizon never
        materializes; fall back to the `("gap", None)` shape so the
        Joiner still fails closed via its own monotonic clock.
        """
        if not self._pending_gap:
            return []
        self._pending_gap = False
        return [("gap", None)]


class ProfileReader:
    """Drain the profile pipe on a background thread; decode on the caller.

    The thread does nothing but blocking os.read into a queue, so the pipe
    empties as fast as the kernel delivers. RecordReader and every consumer
    stay single-threaded on the caller (the main event loop). The queue is
    bounded: if the caller falls behind, the pipe fills and the target drops
    and counts records instead of growing this process without limit.
    """

    def __init__(self, fd: int, max_chunks: int = _MAX_QUEUED_CHUNKS):
        # Own a PRIVATE dup of the read end. The main thread may close the
        # original profile_r (e.g. _cleanup_before_release); closing an fd
        # under a blocked os.read is undefined, and the number could be
        # reused. The dup shares the pipe's open file description, so it still
        # sees EOF when the child closes the write end, and the reader thread
        # closes only its own dup.
        self._fd = os.dup(fd)
        self._q: queue.Queue = queue.Queue(maxsize=max_chunks)
        self._reader = RecordReader()
        self._thread = threading.Thread(target=self._run, name="metagross-profile",
                                        daemon=True)
        self.at_eof = False

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                data = os.read(self._fd, 65536)  # blocking
                self._q.put(data)
                if not data:
                    break  # EOF: write end closed
        except OSError:
            self._q.put(b"")  # surface as EOF; never raise off-thread
        finally:
            try:
                os.close(self._fd)  # close our own dup only
            except OSError:
                pass
    # If a grandchild the target forked keeps the write end open past the
    # final-drain deadline, this thread stays blocked in os.read on its dup;
    # the daemon thread and its dup are reclaimed at process exit, which is
    # imminent once run_live returns. Do not close the dup from another thread.

    def _consume(self, item) -> list:
        if not item:
            self.at_eof = True
            return self._reader.finalize()
        return self._reader.feed(item)

    def lost_records(self) -> int:
        """Return the reader's total lost-record count (sum of gap deltas).

        This is the authoritative loss count for the summary: the writer's
        own dropped-count exists only for unit-testing the writer in
        isolation (see `_DropCountWriter`).
        """
        return self._reader.lost_records

    def hook_replacements(self) -> int:
        """Return how often the target replaced the profiling hook."""
        return self._reader.hook_replacements

    def poll(self) -> list:
        out = []
        while not self.at_eof:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            out.extend(self._consume(item))
        if self._reader._pending_gap:
            # A gap revealed on a no-ts record is normally resolved by the
            # next real-ts record (see RecordReader._note_seq), but poll()
            # is called roughly every event-loop tick and the resolving
            # record may not have arrived yet even under normal operation.
            # Left unresolved across ticks, a pre-gap CALL whose RETURN was
            # ALSO dropped under the same overrun could sit in the timeline
            # long enough for Joiner.flush's >100ms hold window to release
            # a GPU event against it -- exactly the leaked-frame
            # mis-attribution on_gap exists to prevent. Flush it now as
            # ("gap", None) rather than risk that: intentionally
            # over-conservative (a resolving record merely in flight across
            # a chunk boundary gets ("gap", now) for one flush), but only
            # under overrun, and never a guessed frame either way.
            out.extend(self._reader.finalize())
        return out

    def drain_to_eof(self, deadline: float) -> list:
        out = []
        while not self.at_eof:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self._q.get(timeout=remaining)
            except queue.Empty:
                break
            out.extend(self._consume(item))
        if not self.at_eof:
            # Timed out waiting for real EOF (e.g. a forked grandchild
            # still holds the write end open past the final-drain
            # deadline). This is still the terminal drain call before the
            # joiner's final forced flush, so a gap still pending -- one
            # revealed on a no-ts record with no later real-ts record to
            # resolve it -- must be flushed now rather than left to
            # silently vanish (fail-closed).
            out.extend(self._reader.finalize())
        return out


_EXCLUDED_PARTS = ("site-packages", "dist-packages")
_SELF_DIR = os.path.dirname(os.path.realpath(__file__))


class _ProjectClassifier:
    """Cache project membership and static record fields for profile events."""

    def __init__(self, project_root: str):
        self.root = os.path.realpath(project_root)
        self.root_prefix = (self.root if self.root.endswith(os.sep)
                            else self.root + os.sep)
        self._path_cache: dict[str, bool] = {}
        self._code_cache: dict[object, tuple[str, str, int] | None] = {}

    def includes(self, path: str) -> bool:
        cached = self._path_cache.get(path)
        if cached is not None:
            return cached
        real = os.path.realpath(path)
        # "<string>", "<frozen ...>" and similar names belong to code with no
        # source file; resolved as a path they would land under the cwd.
        included = (not path.startswith("<")
                    and real.startswith(self.root_prefix))
        if included and real.startswith(_SELF_DIR + os.sep):
            included = False
        if included:
            relative = real[len(self.root_prefix):]
            included = not any(
                part in _EXCLUDED_PARTS for part in relative.split(os.sep)
            )
        self._path_cache[path] = included
        return included

    def metadata(self, code) -> tuple[str, str, int] | None:
        """Return the (func, path, line) key `_FrameEmitter` interns on.

        Cached as plain strings, not pre-encoded bytes: with interning, a
        repeat frame only ever needs this tuple for the dict lookup, and
        the utf-8 encode + `_MAX_STR` truncation (in `_encode_frame_def`)
        runs at most once per unique frame, the first time it is seen.
        """
        try:
            return self._code_cache[code]
        except KeyError:
            pass
        if not self.includes(code.co_filename):
            self._code_cache[code] = None
            return None
        metadata = (code.co_name, code.co_filename, code.co_firstlineno)
        self._code_cache[code] = metadata
        return metadata


def is_project_file(path: str, project_root: str) -> bool:
    return _ProjectClassifier(project_root).includes(path)


_current_emitter: "_FrameEmitter | None" = None


def current_emitter() -> "_FrameEmitter | None":
    """Return the installed `_FrameEmitter`, or `None` if profiling is not
    installed (or has been disabled, e.g. in a forked child).
    """
    return _current_emitter


class OpenSpan:
    """One `metagross.span()` block; `name` becomes None once it closes."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


# The spans open in the running context, outermost first. A context variable
# gives every thread and every asyncio task its own stack.
open_spans: contextvars.ContextVar[tuple[OpenSpan, ...]] = (
    contextvars.ContextVar("metagross_open_spans", default=()))
_reported_span = threading.local()  # the span last written for this thread
_spans_used = False


def report_span() -> None:
    """Write this thread's active span if it changed since the last report.

    `span()` calls this when a block opens and closes. From the first span
    on, the profiling hook also calls it whenever the running context's
    innermost span differs from the last report, which is how a switch
    between asyncio tasks reaches the tracer.
    """
    global _spans_used
    emitter = _current_emitter
    if emitter is None:
        return
    _spans_used = True
    stack = open_spans.get()
    name = stack[-1].name if stack else None
    if name != getattr(_reported_span, "name", None):
        # Record the change before writing it: the write makes Python calls,
        # and the hook must find nothing left to report from inside them.
        _reported_span.name = name
        emitter.span(threading.get_native_id(), time.monotonic_ns(), name)


def install(write_fd: int, project_root: str) -> None:
    global _current_emitter, _spans_used
    _spans_used = False
    local = threading.local()
    classifier = _ProjectClassifier(project_root)

    try:
        os.write(write_fd, encode_hello(os.getpid(), time.monotonic_ns()))
    except OSError:
        pass  # broken trace pipe must never kill the target

    # Only after HELLO is safely on the wire does the fd go non-blocking:
    # an EAGAIN on HELLO itself would desync the reader's very first
    # expected seq before anything downstream can recover from it.
    os.set_blocking(write_fd, False)
    writer = _DropCountWriter(write_fd)
    emitter = _FrameEmitter(writer.write)
    _current_emitter = emitter

    def hook(frame, event, arg):
        if event == "call":
            kind = CALL
            if _spans_used:
                # A resumed asyncio task starts with a call event in its own
                # context, and any Python frame can be that first one.
                stack = open_spans.get()
                if ((stack[-1].name if stack else None)
                        != getattr(_reported_span, "name", None)):
                    report_span()
        elif event == "return":
            kind = RETURN
        else:
            return
        metadata = classifier.metadata(frame.f_code)
        if metadata is None:
            return
        tid = getattr(local, "tid", None)
        if tid is None:
            tid = local.tid = threading.get_native_id()
        func, path, line = metadata
        ts_ns = time.monotonic_ns()
        # threading.setprofile fires this hook on every target thread; the
        # seq assignment and the write(s) that carry it must stay ordered
        # together, or the reader sees seqs out of order and manufactures
        # false gaps (see _next_seq's docstring). emit() may issue two
        # writes (a first-sight FRAME_DEF, then the CALL/RETURN) that must
        # land back-to-back -- and a span report on another thread must
        # not interleave with either. `emitter`'s own
        # lock (held inside `emit`) covers both concerns; there is no
        # separate lock here to take.
        emitter.emit(kind, tid, ts_ns, func, path, line)

    def _disable_in_forked_child() -> None:
        # A target that forks without exec (e.g. multiprocessing/DataLoader
        # workers) inherits this hook, the pipe fd, and this module's seq
        # counter at its current value. eBPF only tracks the original
        # tgid's GPU calls, so a forked child's own profile records were
        # never load-bearing for attribution -- but if it kept emitting
        # them into the SAME pipe with a copied (colliding) seq sequence,
        # the reader would see interleaved/duplicate seqs and manufacture
        # continuous false gaps, corrupting the PARENT's real attribution.
        # Stop tracing in the child instead.
        #
        # Deadlock note: this function's own entry fires the still-active
        # `hook` as a profile "call" event before the body below runs and
        # clears it. That is safe only because `_ProjectClassifier` excludes
        # every file under this package's own directory (_SELF_DIR), so
        # `hook` returns before reaching `emitter.emit()` -- and its
        # internal `emitter._lock` -- for any frame defined in _profile.py,
        # including this one. Do not relax that self-exclusion, or a lock
        # some other thread held at the moment of fork (and so is held
        # forever in this single-threaded child) would deadlock right here.
        global _current_emitter
        _current_emitter = None  # first: `audit` must not write from here
        sys.setprofile(None)
        threading.setprofile(None)
        # AGENTS.md: be conservative with inherited file descriptors. A
        # long-lived non-exec worker holding the write end open would
        # otherwise delay the parent's EOF until the final-drain deadline.
        try:
            os.close(write_fd)
        except OSError:
            pass

    os.register_at_fork(after_in_child=_disable_in_forked_child)

    def audit(event, args):
        # sys.setprofile is audited before it takes effect, from Python and
        # from the C API alike. If the hook is still this thread's profile
        # function, the target is about to replace it.
        if (event == "sys.setprofile" and _current_emitter is emitter
                and sys.getprofile() is hook):
            emitter.hook_replaced(time.monotonic_ns())

    threading.setprofile(hook)
    sys.setprofile(hook)
    sys.addaudithook(audit)
