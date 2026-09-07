# metagross/_follow.py
"""Incremental, bounded readers for live Metagross trace dashboards."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from metagross import _viewer


_READ_CHUNK_BYTES = 64 << 10
_MAX_BYTES_PER_POLL = 4 << 20
_PREFIX_BYTES = 256


@dataclasses.dataclass(frozen=True)
class FollowUpdate:
    events: int = 0
    malformed_lines: int = 0
    reset: bool = False
    waiting: bool = False


class TraceFollower:
    """Follow one JSONL path while bounding partial records and stored state."""

    def __init__(self, path: Path, recent_limit: int = _viewer._DEFAULT_RECENT):
        self.path = path
        self.recent_limit = recent_limit
        self.model = _viewer.TraceModel(recent_limit=recent_limit)
        self._stream = None
        self._identity: tuple[int, int] | None = None
        self._opened_once = False
        self._partial = bytearray()
        self._discarding_line = False
        self._prefix = bytearray()
        self._tail = bytearray()
        self._observed_size = 0
        self._mtime_ns = 0

    @property
    def trace_mtime_ns(self) -> int:
        return self._mtime_ns

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = None
        self._identity = None

    def _open(self, reset: bool) -> None:
        self.close()
        self._stream = _viewer._open_regular_binary(self.path, "trace")
        try:
            opened_info = os.fstat(self._stream.fileno())
        except OSError as exc:
            self.close()
            raise _viewer.ViewerError(
                f"cannot inspect trace {str(self.path)!r}: {exc}"
            ) from None
        self._identity = (opened_info.st_dev, opened_info.st_ino)
        self._mtime_ns = opened_info.st_mtime_ns
        self._observed_size = opened_info.st_size
        self._partial.clear()
        self._discarding_line = False
        self._prefix.clear()
        self._tail.clear()
        if reset:
            self.model = _viewer.TraceModel(recent_limit=self.recent_limit)
        self._opened_once = True

    def _content_changed(self, info: os.stat_result) -> bool:
        if self._stream is None:
            return False
        consumed_offset = self._stream.tell()
        if info.st_size < consumed_offset:
            return True
        if info.st_size == self._observed_size and info.st_mtime_ns != self._mtime_ns:
            return True
        if info.st_size == self._observed_size and info.st_mtime_ns == self._mtime_ns:
            return False
        try:
            with _viewer._open_regular_binary(self.path, "trace") as stream:
                current_info = os.fstat(stream.fileno())
                if (current_info.st_dev, current_info.st_ino) != self._identity:
                    return True
                if self._prefix:
                    current_prefix = stream.read(len(self._prefix))
                    if current_prefix != bytes(self._prefix):
                        return True
                if self._tail:
                    stream.seek(consumed_offset - len(self._tail))
                    current_tail = stream.read(len(self._tail))
                    if current_tail != bytes(self._tail):
                        return True
        except OSError as exc:
            raise _viewer.ViewerError(
                f"cannot inspect trace {str(self.path)!r}: {exc}"
            ) from None
        return False

    def _feed(self, data: bytes) -> None:
        if self._discarding_line:
            newline = data.find(b"\n")
            if newline < 0:
                return
            data = data[newline + 1 :]
            self._discarding_line = False

        self._partial.extend(data)
        while True:
            newline = self._partial.find(b"\n")
            if newline < 0:
                break
            raw = bytes(self._partial[: newline + 1])
            del self._partial[: newline + 1]
            if len(raw) > _viewer._MAX_LINE_BYTES:
                self.model.malformed_lines += 1
            else:
                _viewer.observe_raw_line(self.model, raw)

        if len(self._partial) > _viewer._MAX_LINE_BYTES:
            self.model.malformed_lines += 1
            self._partial.clear()
            self._discarding_line = True

    def poll(self, max_bytes: int = _MAX_BYTES_PER_POLL) -> FollowUpdate:
        before_events = self.model.events
        before_malformed = self.model.malformed_lines
        reset = False
        try:
            info = self.path.stat()
        except FileNotFoundError:
            if self._stream is not None:
                self.close()
            return FollowUpdate(waiting=True)
        except OSError as exc:
            raise _viewer.ViewerError(
                f"cannot inspect trace {str(self.path)!r}: {exc}"
            ) from None

        identity = (info.st_dev, info.st_ino)
        if self._stream is None:
            reset = self._opened_once
            self._open(reset=reset)
        elif identity != self._identity or self._content_changed(info):
            reset = True
            self._open(reset=True)

        if reset:
            before_events = 0
            before_malformed = 0
        if self._stream is None:
            return FollowUpdate(reset=reset, waiting=True)

        self._mtime_ns = info.st_mtime_ns
        self._observed_size = info.st_size
        remaining = max(0, max_bytes)
        try:
            while remaining:
                chunk = self._stream.read(min(_READ_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                if len(self._prefix) < _PREFIX_BYTES:
                    wanted = _PREFIX_BYTES - len(self._prefix)
                    self._prefix.extend(chunk[:wanted])
                self._tail.extend(chunk)
                if len(self._tail) > _PREFIX_BYTES:
                    del self._tail[:-_PREFIX_BYTES]
                self._feed(chunk)
                remaining -= len(chunk)
        except OSError as exc:
            self.close()
            raise _viewer.ViewerError(
                f"cannot read trace {str(self.path)!r}: {exc}"
            ) from None

        return FollowUpdate(
            events=self.model.events - before_events,
            malformed_lines=self.model.malformed_lines - before_malformed,
            reset=reset,
            waiting=False,
        )


class SummaryFollower:
    """Load a final summary when it appears or changes."""

    def __init__(self, path: Path | None):
        self.path = path
        self.last_error: str | None = None
        self._last_signature: tuple[int, int, int, int] | None = None
        self._ignored_signature: tuple[int, int, int, int] | None = None

    def _signature(self) -> tuple[int, int, int, int] | None:
        if self.path is None:
            return None
        try:
            info = self.path.stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise _viewer.ViewerError(
                f"cannot inspect summary {str(self.path)!r}: {exc}"
            ) from None
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)

    def reset(self, trace_mtime_ns: int) -> None:
        self._last_signature = None
        self._ignored_signature = None
        self.last_error = None
        try:
            signature = self._signature()
        except _viewer.ViewerError as exc:
            self.last_error = str(exc)
            return
        if signature is not None and signature[2] > 0 and signature[3] < trace_mtime_ns:
            self._ignored_signature = signature

    def poll(self, model: _viewer.TraceModel) -> bool:
        try:
            signature = self._signature()
        except _viewer.ViewerError as exc:
            model.clear_summary()
            self.last_error = str(exc)
            return False
        if signature is None:
            model.clear_summary()
            self._last_signature = None
            self._ignored_signature = None
            self.last_error = None
            return False
        if signature == self._ignored_signature:
            model.clear_summary()
            self.last_error = None
            return False
        if signature == self._last_signature:
            return False

        model.clear_summary()
        self._last_signature = signature
        if signature[2] == 0:
            self.last_error = None
            return False
        try:
            summary = _viewer.load_summary(self.path)
            model.load_summary(summary)
        except _viewer.ViewerError as exc:
            self.last_error = str(exc)
            return False
        self.last_error = None
        self._ignored_signature = None
        return True
