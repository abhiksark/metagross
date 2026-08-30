# metagross/_profile.py
"""Profiling hooks run inside the traced child; record codec."""
from __future__ import annotations

import os
import struct
import sys
import threading
import time

CALL, RETURN = 0, 1
_HEADER = struct.Struct("<BIQIHH")
_MAX_STR = 500


def _encode_record_bytes(kind, tid, ts_ns, func_bytes, path_bytes,
                         line) -> bytes:
    return (_HEADER.pack(kind, tid, ts_ns, line, len(func_bytes), len(path_bytes))
            + func_bytes + path_bytes)


def encode_record(kind, tid, ts_ns, func, path, line) -> bytes:
    func_bytes = func.encode("utf-8", "replace")[:_MAX_STR]
    path_bytes = path.encode("utf-8", "replace")[:_MAX_STR]
    return _encode_record_bytes(kind, tid, ts_ns, func_bytes, path_bytes, line)


class RecordReader:
    def __init__(self):
        self._buf = b""

    def feed(self, data: bytes) -> list[tuple]:
        self._buf += data
        out = []
        while len(self._buf) >= _HEADER.size:
            kind, tid, ts, line, fl, pl = _HEADER.unpack_from(self._buf)
            total = _HEADER.size + fl + pl
            if len(self._buf) < total:
                break
            func = self._buf[_HEADER.size:_HEADER.size + fl].decode("utf-8", "replace")
            path = self._buf[_HEADER.size + fl:total].decode("utf-8", "replace")
            self._buf = self._buf[total:]
            out.append((kind, tid, ts, func, path, line))
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
        self._code_cache: dict[object, tuple[bytes, bytes, int] | None] = {}

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

    def metadata(self, code) -> tuple[bytes, bytes, int] | None:
        try:
            return self._code_cache[code]
        except KeyError:
            pass
        if not self.includes(code.co_filename):
            self._code_cache[code] = None
            return None
        metadata = (
            code.co_name.encode("utf-8", "replace")[:_MAX_STR],
            code.co_filename.encode("utf-8", "replace")[:_MAX_STR],
            code.co_firstlineno,
        )
        self._code_cache[code] = metadata
        return metadata


def is_project_file(path: str, project_root: str) -> bool:
    return _ProjectClassifier(project_root).includes(path)


def install(write_fd: int, project_root: str) -> None:
    local = threading.local()
    classifier = _ProjectClassifier(project_root)

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
        func_bytes, path_bytes, line = metadata
        record = _encode_record_bytes(
            kind, tid, time.monotonic_ns(), func_bytes, path_bytes, line
        )
        try:
            os.write(write_fd, record)
        except OSError:
            pass  # broken trace pipe must never kill the target

    threading.setprofile(hook)
    sys.setprofile(hook)
