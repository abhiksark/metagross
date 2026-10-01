# metagross/_bpf.py
"""API table, libcuda symbol resolution, BPF program for Metagross."""
from __future__ import annotations

import ctypes as ct
import ctypes.util
import dataclasses
import os
import subprocess
import sys
from typing import Callable

from metagross import MetagrossError

_SUFFIXES = ["", "_v2", "_v3", "_ptds", "_ptsz", "_v2_ptds", "_v2_ptsz"]


@dataclasses.dataclass(frozen=True)
class Api:
    api_id: int
    base: str
    category: str
    mandatory: bool = False


APIS = [
    Api(1, "cuLaunchKernel", "launch", mandatory=True),
    Api(2, "cuLaunchKernelEx", "launch_ex"),
    Api(3, "cuMemAlloc", "alloc"),
    Api(4, "cuMemAllocAsync", "alloc_async"),
    Api(5, "cuMemFree", "free"),
    Api(6, "cuMemFreeAsync", "free_async"),
    Api(7, "cuMemcpyHtoD", "copy_h2d"),
    Api(8, "cuMemcpyHtoDAsync", "copy_h2d"),
    Api(9, "cuMemcpyDtoH", "copy_d2h"),
    Api(10, "cuMemcpyDtoHAsync", "copy_d2h"),
    Api(11, "cuMemcpyDtoD", "copy_d2d"),
    Api(12, "cuMemcpyDtoDAsync", "copy_d2d"),
    Api(13, "cuMemcpy", "copy_generic"),
    Api(14, "cuMemcpyAsync", "copy_generic"),
    Api(15, "cuStreamSynchronize", "sync"),
    Api(16, "cuCtxSynchronize", "sync"),
    Api(17, "cuEventSynchronize", "sync"),
    Api(18, "cuModuleGetFunction", "register"),
    Api(19, "cuLibraryGetKernel", "register"),
    Api(20, "cuKernelGetFunction", "register"),
    Api(21, "cuGraphLaunch", "graph_launch"),
]

API_BY_ID = {a.api_id: a for a in APIS}

_TRACE_CATEGORY_MAP = {
    "launch": frozenset(("launch", "launch_ex", "graph_launch", "register")),
    "memory": frozenset(("alloc", "alloc_async", "free", "free_async")),
    "copy": frozenset(("copy_h2d", "copy_d2h", "copy_d2d", "copy_generic")),
    "sync": frozenset(("sync",)),
}


def select_apis(families: frozenset[str] | None = None) -> list[Api]:
    """Return APIs needed by a user-visible capture-family selection.

    ``None`` means all APIs. Kernel-name registration is an internal dependency
    of launch capture and is therefore selected with the launch family.
    """
    if families is None:
        return list(APIS)
    unknown = families - _TRACE_CATEGORY_MAP.keys()
    if unknown:
        raise MetagrossError(f"unknown trace family: {sorted(unknown)[0]}")
    categories = set()
    for family in families:
        categories.update(_TRACE_CATEGORY_MAP[family])
    return [api for api in APIS if api.category in categories]


@dataclasses.dataclass(frozen=True)
class Attachment:
    symbol: str
    api: Api


def candidate_symbols(base: str) -> list[str]:
    return [base + s for s in _SUFFIXES]


def resolve_attachments(apis, resolver: Callable[[str], int | None]):
    attachments = []
    for api in apis:
        seen_addrs = set()
        found = False
        for sym in candidate_symbols(api.base):
            addr = resolver(sym)
            if addr is None or addr in seen_addrs:
                continue
            seen_addrs.add(addr)
            attachments.append(Attachment(sym, api))
            found = True
        if api.mandatory and not found:
            raise MetagrossError(
                f"mandatory symbol {api.base} not found in libcuda")
    return attachments


def dlsym_resolver(lib_path: str) -> Callable[[str], int | None]:
    lib = ct.CDLL(lib_path)

    def resolve(symbol: str) -> int | None:
        try:
            return ct.cast(getattr(lib, symbol), ct.c_void_p).value
        except AttributeError:
            return None

    return resolve


_LIBCUDA_CANDIDATES = [
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib/libcuda.so.1",
]


def _ldconfig_libcuda_path() -> str | None:
    """Ask ldconfig's cache for libcuda.so.1, preferring the x86-64 entry.

    Last-resort lookup for nonstandard installs where libcuda.so.1 lives
    outside the usual multiarch directories. Any failure (ldconfig missing,
    not executable, unexpected output) is treated as "no result" rather
    than propagated.
    """
    try:
        out = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True,
            check=True, timeout=5).stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    candidates = []
    for line in out.splitlines():
        if "libcuda.so.1" not in line or "=> " not in line:
            continue
        path = line.rsplit("=> ", 1)[-1].strip()
        candidates.append((("x86-64" in line), path))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def find_libcuda() -> str:
    """Locate libcuda.so.1 as an absolute filesystem path.

    An absolute path is required (not merely a soname like "libcuda.so.1")
    because BCC's uprobe attachment opens and parses the target's ELF file
    directly to resolve symbol addresses; it does not go through the
    dynamic loader's search path and cannot resolve a bare soname. ctypes
    and the dynamic loader tolerate a bare soname, which is why that used
    to look like it worked -- but BCC attach_uprobe fails on it.
    """
    path = ctypes.util.find_library("cuda")
    if path is not None and path.startswith("/"):
        return path
    for candidate in _LIBCUDA_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    ldconfig_path = _ldconfig_libcuda_path()
    if ldconfig_path is not None and os.path.exists(ldconfig_path):
        return ldconfig_path
    raise MetagrossError(
        "libcuda.so.1 not found; is the NVIDIA driver installed?")



def mapped_libcuda(maps_text: str) -> set[str]:
    """Return the libcuda files named in a `/proc/<pid>/maps` listing."""
    paths = set()
    for line in maps_text.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and os.path.basename(fields[5]).startswith("libcuda.so"):
            paths.add(fields[5])
    return paths


def target_loaded_libcuda(pid: int, lib_path: str) -> bool:
    """Return True once the target has loaded a libcuda.

    Warn if it is not the file the probes are on: the loader may pick another
    copy than `find_libcuda` did, and uprobes only fire for the probed file.
    """
    try:
        with open(f"/proc/{pid}/maps") as maps:
            mapped = mapped_libcuda(maps.read())
    except OSError:
        return False
    probed = os.path.realpath(lib_path)
    if mapped and probed not in mapped:
        print(f"metagross: the script loaded {sorted(mapped)[0]}, but the "
              f"probes are on {probed}; its CUDA calls are not traced",
              file=sys.stderr)
    return bool(mapped)


_HEADER = r"""
#include <uapi/linux/ptrace.h>

#define NAME_MAX_LEN 1024

struct inflight_t {
    u64 ts;
    u32 api_id;
    u64 args[9];
};

struct event_t {
    u64 ts;
    u64 dur;
    u32 tid;
    u32 api_id;
    s32 ret;
    u32 _pad;
    u64 args[9];
    u64 out;
};

// Sent only by the probes that read a kernel name: the same fields, then
// the name. Keeping the name out of every other event leaves the ring
// buffer room for about twice as many of them.
struct name_event_t {
    struct event_t event;
    char name[NAME_MAX_LEN];
};

BPF_HASH(inflight, u32, struct inflight_t);
BPF_RINGBUF_OUTPUT(events, 1024);
BPF_ARRAY(counters, u64, 2);

static __always_inline void bump(int idx) {
    u64 *v = counters.lookup(&idx);
    if (v) __sync_fetch_and_add(v, 1);
}
"""

_ENTER = r"""
int enter_{base}(struct pt_regs *ctx) {{
    u64 id = bpf_get_current_pid_tgid();
    if ((u32)(id >> 32) != {tgid}) return 0;
    u32 tid = (u32)id;
    struct inflight_t f = {{}};
    f.ts = bpf_ktime_get_ns();
    f.api_id = {api_id};
    f.args[0] = PT_REGS_PARM1(ctx);
    f.args[1] = PT_REGS_PARM2(ctx);
    f.args[2] = PT_REGS_PARM3(ctx);
    f.args[3] = PT_REGS_PARM4(ctx);
    f.args[4] = PT_REGS_PARM5(ctx);
    f.args[5] = PT_REGS_PARM6(ctx);
{enter_extra}
    struct inflight_t *prev = inflight.lookup(&tid);
    if (prev) bump(1);
    inflight.update(&tid, &f);
    return 0;
}}
"""

_EXIT = r"""
int exit_{base}(struct pt_regs *ctx) {{
    u64 id = bpf_get_current_pid_tgid();
    if ((u32)(id >> 32) != {tgid}) return 0;
    u32 tid = (u32)id;
    struct inflight_t *f = inflight.lookup(&tid);
    if (!f) return 0;
    if (f->api_id != {api_id}) {{ return 0; }}
    {reserve}
    if (!e) {{ bump(0); inflight.delete(&tid); return 0; }}
    e->ts = f->ts;
    e->dur = bpf_ktime_get_ns() - f->ts;
    e->tid = tid;
    e->api_id = f->api_id;
    e->ret = (s32)PT_REGS_RC(ctx);
    e->_pad = 0;
    __builtin_memcpy(e->args, f->args, sizeof(e->args));
    e->out = 0;
{exit_extra}
    events.ringbuf_submit({record}, 0);
    inflight.delete(&tid);
    return 0;
}}
"""

# args 7-9 of cuLaunchKernel live on the user stack: [sp+8]=blockDimZ,
# [sp+16]=sharedMemBytes, [sp+24]=hStream (SysV AMD64, 11-arg call).
_LAUNCH_STACK_READS = r"""
    {
        u64 sp = PT_REGS_SP(ctx);
        bpf_probe_read_user(&f.args[6], sizeof(u64), (void *)(sp + 8));
        bpf_probe_read_user(&f.args[7], sizeof(u64), (void *)(sp + 16));
        bpf_probe_read_user(&f.args[8], sizeof(u64), (void *)(sp + 24));
    }
"""

# CUlaunchConfig field offsets (CUDA 12): grid 0/4/8, block 12/16/20,
# sharedMemBytes 24, hStream 32 (u64 after 4-byte pad).
_LAUNCH_EX_CONFIG_READS = r"""
    {
        void *cfg = (void *)f.args[0];
        u32 dims[6] = {};
        u32 shmem = 0; u64 stream = 0;
        bpf_probe_read_user(&dims, sizeof(dims), cfg);
        bpf_probe_read_user(&shmem, sizeof(shmem), cfg + 24);
        bpf_probe_read_user(&stream, sizeof(stream), cfg + 32);
        f.args[2] = ((u64)dims[1] << 32) | dims[0];
        f.args[3] = ((u64)dims[3] << 32) | dims[2];
        f.args[4] = ((u64)dims[5] << 32) | dims[4];
        f.args[5] = shmem;
        f.args[6] = stream;
    }
"""

_READ_OUT_PARAM = r"""
    if (e->ret == 0)
        bpf_probe_read_user(&e->out, sizeof(u64), (void *)f->args[0]);
"""

_READ_OUT_AND_NAME = _READ_OUT_PARAM + r"""
    named->name[0] = 0;
    if (e->ret == 0 && f->args[2])
        bpf_probe_read_user_str(&named->name, NAME_MAX_LEN, (void *)f->args[2]);
"""

_RESERVE_EVENT = (
    "struct event_t *e = events.ringbuf_reserve(sizeof(struct event_t));")
_RESERVE_NAME_EVENT = (
    "struct name_event_t *named ="
    " events.ringbuf_reserve(sizeof(struct name_event_t));\n"
    "    struct event_t *e = named ? &named->event : 0;")

_ENTER_EXTRA = {
    "launch": _LAUNCH_STACK_READS,
    "launch_ex": _LAUNCH_EX_CONFIG_READS,
}
_EXIT_EXTRA = {
    "alloc": _READ_OUT_PARAM,
    "alloc_async": _READ_OUT_PARAM,
    "register": _READ_OUT_PARAM,
}


def build_source(target_tgid: int, apis: list[Api] | None = None) -> str:
    selected = APIS if apis is None else apis
    parts = [_HEADER]
    for api in selected:
        exit_extra = _EXIT_EXTRA.get(api.category, "")
        reserve, record = _RESERVE_EVENT, "e"
        if api.base in ("cuModuleGetFunction", "cuLibraryGetKernel"):
            exit_extra = _READ_OUT_AND_NAME
            reserve, record = _RESERVE_NAME_EVENT, "named"
        parts.append(_ENTER.format(
            base=api.base, tgid=target_tgid, api_id=api.api_id,
            enter_extra=_ENTER_EXTRA.get(api.category, "")))
        parts.append(_EXIT.format(
            base=api.base, tgid=target_tgid, api_id=api.api_id,
            reserve=reserve, record=record, exit_extra=exit_extra))
    return "".join(parts)


class RawEvent(ct.Structure):
    _fields_ = [
        ("ts", ct.c_uint64), ("dur", ct.c_uint64),
        ("tid", ct.c_uint32), ("api_id", ct.c_uint32),
        ("ret", ct.c_int32), ("_pad", ct.c_uint32),
        ("args", ct.c_uint64 * 9), ("out", ct.c_uint64),
        ("name", ct.c_char * 1024),
    ]


def decode_event(data: bytes) -> RawEvent:
    """Decode either record the probes send; most carry no name field."""
    return RawEvent.from_buffer_copy(data.ljust(ct.sizeof(RawEvent), b"\0"))
