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


def encode_record(kind, tid, ts_ns, func, path, line) -> bytes:
    fb = func.encode("utf-8", "replace")[:_MAX_STR]
    pb = path.encode("utf-8", "replace")[:_MAX_STR]
    return _HEADER.pack(kind, tid, ts_ns, line, len(fb), len(pb)) + fb + pb


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


def is_project_file(path: str, project_root: str) -> bool:
    real = os.path.realpath(path)
    root = os.path.realpath(project_root)
    if not real.startswith(root + os.sep):
        return False
    if real.startswith(_SELF_DIR + os.sep):
        return False
    parts = real[len(root):].split(os.sep)
    return not any(p in _EXCLUDED_PARTS for p in parts)


def install(write_fd: int, project_root: str) -> None:
    local = threading.local()

    def hook(frame, event, arg):
        if event == "call":
            kind = CALL
        elif event == "return":
            kind = RETURN
        else:
            return
        code = frame.f_code
        if not is_project_file(code.co_filename, project_root):
            return
        tid = getattr(local, "tid", None)
        if tid is None:
            tid = local.tid = threading.get_native_id()
        record = encode_record(kind, tid, time.monotonic_ns(),
                               code.co_name, code.co_filename,
                               code.co_firstlineno)
        try:
            os.write(write_fd, record)
        except OSError:
            pass  # broken trace pipe must never kill the target

    threading.setprofile(hook)
    sys.setprofile(hook)
