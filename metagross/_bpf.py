# metagross/_bpf.py
"""API table, libcuda symbol resolution, BPF program for Metagross."""
from __future__ import annotations

import ctypes
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
    lib = ctypes.CDLL(lib_path)

    def resolve(symbol: str) -> int | None:
        try:
            return ctypes.cast(getattr(lib, symbol), ctypes.c_void_p).value
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
