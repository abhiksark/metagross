# metagross/_events.py
"""Event enrichment, attribution, and rendering for Metagross."""
from __future__ import annotations

import shlex


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


def _hex(v: int) -> str:
	return f"0x{v:x}"


def describe(api, ev, registry: KernelRegistry, allocs: AllocTracker):
	cat = api.category
	kernel = None
	det: dict = {}
	if cat in ("launch", "launch_ex"):
		if cat == "launch":
			handle = ev.args[0]
			det["grid"] = f"{ev.args[1]},{ev.args[2]},{ev.args[3]}"
			det["block"] = f"{ev.args[4]},{ev.args[5]},{ev.args[6] & 0xFFFFFFFF}"
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
