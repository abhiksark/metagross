# metagross/_profile.py
"""Profiling hooks run inside the traced child; record codec (wire v2)."""
from __future__ import annotations

import os
import queue
import struct
import sys
import threading
import time

# Frame "kind" values used by FrameTimeline.on_record and everywhere a
# decoded frame record is consumed. These are UNCHANGED from Phase A: they
# are not the wire rtype (see below), just call-vs-return direction.
CALL, RETURN = 0, 1

# Wire record types (the `rtype` byte in the common header). CALL and RETURN
# here are deliberately not reused as names: `_FRAME_CALL_RTYPE` /
# `_FRAME_RETURN_RTYPE` carry the CALL/RETURN distinction on the wire, kept
# private because nothing outside this module needs the raw byte value; the
# public `CALL`/`RETURN` kind constants above must stay 0/1 for Phase A
# compatibility (FrameTimeline.on_record and the many tests that call it
# directly).
HELLO = 0
FRAME_DEF = 1
_FRAME_CALL_RTYPE = 2
_FRAME_RETURN_RTYPE = 3
SPAN_BEGIN = 4
SPAN_END = 5

_WIRE_VERSION = 2
_SEQ_MOD = 2**32
_MAX_STR = 500

# Every record starts with this 6-byte common header.
_COMMON = struct.Struct("<BIB")  # rtype, seq, _reserved
_HELLO_BODY = struct.Struct("<BIQ")  # version, pid, start_ns
_FRAME_DEF_PREFIX = struct.Struct("<IIIHH")  # frame_id, line, _pad, func_len, path_len
# CALL/RETURN carry only the interned frame_id; FRAME_DEF (above) is what
# assigns func/path/line to that id, once, the first time a frame is seen.
_FRAME_REF_BODY = struct.Struct("<IQI")  # tid, ts_ns, frame_id
_SPAN_BEGIN_PREFIX = struct.Struct("<IQH")  # tid, ts_ns, name_len
_SPAN_END_BODY = struct.Struct("<IQ")  # tid, ts_ns

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
    """

    def __init__(self, write):
        self._write = write
        self._frames: dict[tuple[str, str, int], int] = {}
        self._next_id = 0

    def emit(self, kind, tid, ts, func, path, line) -> None:
        key = (func, path, line)
        frame_id = self._frames.get(key)
        if frame_id is None:
            frame_id = self._next_id
            self._next_id += 1
            self._frames[key] = frame_id
            self._write(_encode_frame_def(frame_id, func, path, line))
        self._write(_encode_frame_ref(kind, tid, ts, frame_id))


def encode_span_begin(tid, ts_ns, name) -> bytes:
    name_bytes = name.encode("utf-8", "replace")[:_MAX_STR]
    seq = _next_seq()
    return (_COMMON.pack(SPAN_BEGIN, seq, 0)
            + _SPAN_BEGIN_PREFIX.pack(tid, ts_ns, len(name_bytes)) + name_bytes)


def encode_span_end(tid, ts_ns) -> bytes:
    seq = _next_seq()
    return _COMMON.pack(SPAN_END, seq, 0) + _SPAN_END_BODY.pack(tid, ts_ns)


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
        # gap: FRAME_DEF is never dropped (Task 4 makes it undroppable), so
        # a definition always precedes its use and ids never desync.
        self._frames: dict[int, tuple] = {}
        # Defensive-only counter: a CALL/RETURN whose frame_id is somehow
        # unknown is dropped rather than guessing a frame. This should not
        # happen once Task 4 lands (FRAME_DEF cannot itself be lost).
        self.lost_records = 0

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
                self._expected = (seq + 1) % _SEQ_MOD
                self._buf = self._buf[need:]
                continue
            if rtype == FRAME_DEF:
                prefix_off = _COMMON.size
                if len(self._buf) < prefix_off + _FRAME_DEF_PREFIX.size:
                    break
                frame_id, line, _pad, fl, pl = _FRAME_DEF_PREFIX.unpack_from(
                    self._buf, prefix_off)
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
                    # Defensive only: FRAME_DEF is never dropped (Task 4),
                    # so this should not happen. Fail closed -- drop this
                    # one record and count it, never guess a frame, and
                    # never clear the map on the strength of one bad id.
                    self.lost_records += 1
                    continue
                kind = _KIND_BY_FRAME_RTYPE[rtype]
                out.append(("frame", kind, tid, ts_ns, *frame))
                continue
            if rtype == SPAN_BEGIN:
                prefix_off = _COMMON.size
                if len(self._buf) < prefix_off + _SPAN_BEGIN_PREFIX.size:
                    break
                tid, ts_ns, nl = _SPAN_BEGIN_PREFIX.unpack_from(
                    self._buf, prefix_off)
                body_off = prefix_off + _SPAN_BEGIN_PREFIX.size
                total = body_off + nl
                if len(self._buf) < total:
                    break
                name = self._buf[body_off:total].decode("utf-8", "replace")
                self._buf = self._buf[total:]
                self._note_seq(seq, out, ts_ns)
                out.append(("span_begin", tid, ts_ns, name))
                continue
            if rtype == SPAN_END:
                need = _COMMON.size + _SPAN_END_BODY.size
                if len(self._buf) < need:
                    break
                tid, ts_ns = _SPAN_END_BODY.unpack_from(self._buf, _COMMON.size)
                self._buf = self._buf[need:]
                self._note_seq(seq, out, ts_ns)
                out.append(("span_end", tid, ts_ns))
                continue
            # Unknown rtype: there is no length field we can trust, so the
            # stream cannot be resynchronized. Drop it and fail closed
            # rather than raise or spin forever on the same bytes.
            self._buf = b""
            out.append(("gap", None))
            break
        return out

    def _note_seq(self, seq: int, out: list, ts_ns) -> None:
        if self._expected is None or seq != self._expected:
            out.append(("gap", ts_ns))
        self._expected = (seq + 1) % _SEQ_MOD


class ProfileReader:
    """Drain the profile pipe on a background thread; decode on the caller.

    The thread does nothing but blocking os.read into a queue, so the pipe
    empties as fast as the kernel delivers. RecordReader and every consumer
    stay single-threaded on the caller (the main event loop).
    """

    def __init__(self, fd: int):
        # Own a PRIVATE dup of the read end. The main thread may close the
        # original profile_r (e.g. _cleanup_before_release); closing an fd
        # under a blocked os.read is undefined, and the number could be
        # reused. The dup shares the pipe's open file description, so it still
        # sees EOF when the child closes the write end, and the reader thread
        # closes only its own dup.
        self._fd = os.dup(fd)
        self._q: queue.SimpleQueue = queue.SimpleQueue()
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
            return []
        return self._reader.feed(item)

    def poll(self) -> list:
        out = []
        while not self.at_eof:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            out.extend(self._consume(item))
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
        included = real.startswith(self.root_prefix)
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


def install(write_fd: int, project_root: str) -> None:
    local = threading.local()
    classifier = _ProjectClassifier(project_root)
    lock = threading.Lock()

    def _write(record: bytes) -> None:
        try:
            os.write(write_fd, record)
        except OSError:
            pass  # broken trace pipe must never kill the target

    emitter = _FrameEmitter(_write)

    _write(encode_hello(os.getpid(), time.monotonic_ns()))

    def hook(frame, event, arg):
        if event == "call":
            kind = CALL
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
        # land back-to-back, so both happen under the same lock.
        with lock:
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
        # `hook` returns before reaching `with lock:` for any frame defined
        # in _profile.py -- including this one. Do not relax that
        # self-exclusion, or a `lock` some other thread held at the moment
        # of fork (and so is held forever in this single-threaded child)
        # would deadlock right here.
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

    threading.setprofile(hook)
    sys.setprofile(hook)
