# metagross/_bpf.py
"""API table, libcuda symbol resolution, BPF program for Metagross."""
from __future__ import annotations

import ctypes as ct
import ctypes.util
import dataclasses
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
]

API_BY_ID = {a.api_id: a for a in APIS}


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


def find_libcuda() -> str:
    path = ctypes.util.find_library("cuda")
    if path is None:
        fallback = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
        import os
        if os.path.exists(fallback):
            return fallback
        raise MetagrossError(
            "libcuda.so.1 not found; is the NVIDIA driver installed?")
    return path


_HEADER = r"""
#include <uapi/linux/ptrace.h>

#define NAME_MAX_LEN 128

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
    char name[NAME_MAX_LEN];
};

BPF_HASH(inflight, u32, struct inflight_t);
BPF_RINGBUF_OUTPUT(events, 256);
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
    struct event_t *e = events.ringbuf_reserve(sizeof(struct event_t));
    if (!e) {{ bump(0); inflight.delete(&tid); return 0; }}
    e->ts = f->ts;
    e->dur = bpf_ktime_get_ns() - f->ts;
    e->tid = tid;
    e->api_id = f->api_id;
    e->ret = (s32)PT_REGS_RC(ctx);
    e->_pad = 0;
    __builtin_memcpy(e->args, f->args, sizeof(e->args));
    e->out = 0;
    e->name[0] = 0;
{exit_extra}
    events.ringbuf_submit(e, 0);
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
    if (e->ret == 0 && f->args[2])
        bpf_probe_read_user_str(&e->name, NAME_MAX_LEN, (void *)f->args[2]);
"""

_ENTER_EXTRA = {
    "launch": _LAUNCH_STACK_READS,
    "launch_ex": _LAUNCH_EX_CONFIG_READS,
}
_EXIT_EXTRA = {
    "alloc": _READ_OUT_PARAM,
    "alloc_async": _READ_OUT_PARAM,
    "register": _READ_OUT_PARAM,
}


def build_source(target_tgid: int) -> str:
    parts = [_HEADER]
    for api in APIS:
        exit_extra = _EXIT_EXTRA.get(api.category, "")
        if api.base in ("cuModuleGetFunction", "cuLibraryGetKernel"):
            exit_extra = _READ_OUT_AND_NAME
        parts.append(_ENTER.format(
            base=api.base, tgid=target_tgid, api_id=api.api_id,
            enter_extra=_ENTER_EXTRA.get(api.category, "")))
        parts.append(_EXIT.format(
            base=api.base, tgid=target_tgid, api_id=api.api_id,
            exit_extra=exit_extra))
    return "".join(parts)


class RawEvent(ct.Structure):
    _fields_ = [
        ("ts", ct.c_uint64), ("dur", ct.c_uint64),
        ("tid", ct.c_uint32), ("api_id", ct.c_uint32),
        ("ret", ct.c_int32), ("_pad", ct.c_uint32),
        ("args", ct.c_uint64 * 9), ("out", ct.c_uint64),
        ("name", ct.c_char * 128),
    ]


def decode_event(data: bytes) -> RawEvent:
    return RawEvent.from_buffer_copy(data)
