# tests/test_stub_live.py
"""Live tracing against a stub libcuda: needs root and BCC, but no GPU.

Enabled by RUN_STUB_INTEGRATION=1. Build tests/stub_libcuda.c and install it
as the libcuda.so.1 that Metagross finds, on a machine with no NVIDIA driver:

    gcc -O0 -shared -fPIC -o /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
        tests/stub_libcuda.c

Then run this module as root. It starts the real tracer, so it loads BPF
probes; use a disposable machine or container.
"""
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from metagross import MetagrossError
from metagross import _bpf

ENABLED = os.environ.get("RUN_STUB_INTEGRATION") == "1"

_SCRIPT = '''\
import ctypes

cuda = ctypes.CDLL("libcuda.so.1")
cuda.cuLaunchKernel.argtypes = (
    [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3)
cuda.cuGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
cuda.cuLaunchKernelEx.argtypes = [ctypes.c_void_p] * 4
cuda.cuKernelGetFunction.argtypes = [ctypes.c_void_p] * 2
SIZE, STREAM = ctypes.c_size_t, ctypes.c_void_p(0x77)


class LaunchConfig(ctypes.Structure):
    _fields_ = [("grid", ctypes.c_uint * 3), ("block", ctypes.c_uint * 3),
                ("shared_bytes", ctypes.c_uint), ("stream", ctypes.c_void_p),
                ("attributes", ctypes.c_void_p), ("attribute_count", ctypes.c_uint)]


def allocate():
    pointer = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(4096))
    return pointer


def upload(pointer):
    host = ctypes.create_string_buffer(64)
    cuda.cuMemcpyHtoD_v2(pointer, host, ctypes.c_size_t(64))


def launch(function):
    cuda.cuLaunchKernel(function, 8, 1, 1, 128, 1, 2, 48, 0x77, None, None)


def launch_from_library():
    kernel, function = ctypes.c_void_p(), ctypes.c_void_p()
    cuda.cuLibraryGetKernel(ctypes.byref(kernel), None, b"library_kernel")
    cuda.cuKernelGetFunction(ctypes.byref(function), kernel)
    config = LaunchConfig((4, 2, 1), (64, 2, 1), 256, 0x99, None, 0)
    cuda.cuLaunchKernelEx(ctypes.byref(config), function, None, None)


def copy_around(pointer):
    host = ctypes.create_string_buffer(64)
    cuda.cuMemcpyHtoDAsync_v2(pointer, host, SIZE(16), STREAM)
    cuda.cuMemcpyDtoH_v2(host, pointer, SIZE(32))
    cuda.cuMemcpyDtoHAsync_v2(host, pointer, SIZE(8), STREAM)
    cuda.cuMemcpyDtoD_v2(pointer, pointer, SIZE(4))
    cuda.cuMemcpyDtoDAsync_v2(pointer, pointer, SIZE(2), STREAM)
    cuda.cuMemcpy(pointer, pointer, SIZE(1))
    cuda.cuMemcpyAsync(pointer, pointer, SIZE(3), STREAM)


def use_pool():
    pointer = ctypes.c_uint64()
    cuda.cuMemAllocAsync(ctypes.byref(pointer), SIZE(8192), STREAM)
    cuda.cuMemFreeAsync(pointer, STREAM)


def wait_for_stream():
    cuda.cuStreamSynchronize_ptsz(STREAM)
    cuda.cuEventSynchronize(ctypes.c_void_p(0x55))


def replay():
    cuda.cuGraphLaunch(0xABC0, 0x77)


def wait():
    cuda.cuCtxSynchronize()


def release(pointer):
    cuda.cuMemFree_v2(pointer)


def main():
    pointer = allocate()
    upload(pointer)
    function = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(function), None, b"stub_kernel")
    launch(function)
    replay()
    wait()
    launch_from_library()
    copy_around(pointer)
    use_pool()
    wait_for_stream()
    release(pointer)
    print("stub workload done")


main()
'''


def _stub_is_installed() -> bool:
    try:
        library = ctypes.CDLL(_bpf.find_libcuda())
    except (MetagrossError, OSError):
        return False
    return hasattr(library, "metagross_stub_libcuda")


@unittest.skipUnless(ENABLED and os.geteuid() == 0,
                     "needs RUN_STUB_INTEGRATION=1 and root")
class StubLiveTraceTest(unittest.TestCase):
    def setUp(self):
        if not _stub_is_installed():
            self.fail("libcuda.so.1 is missing or is not the stub; see the "
                      "module docstring")
        # Readable by the user the target is dropped to.
        self.project = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.project, ignore_errors=True)
        os.chmod(self.project, 0o755)
        self.script = os.path.join(self.project, "workload.py")
        with open(self.script, "w", encoding="utf-8") as stream:
            stream.write(_SCRIPT)
        os.chmod(self.script, 0o644)
        # Metagross creates the output files itself and hands them to the
        # invoking user; see _fresh_output_path in test_metagross.
        self.out_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.out_dir, ignore_errors=True)

    def _trace(self, *options):
        trace = os.path.join(self.out_dir, "trace.jsonl")
        summary = os.path.join(self.out_dir, "summary.json")
        # Through sudo the target runs as the invoking user; in a bare root
        # shell (a container) there is nobody to drop to.
        root_target = [] if "SUDO_UID" in os.environ else ["--allow-root-target"]
        process = subprocess.run(
            [sys.executable, "-m", "metagross", "--json", "--output", trace,
             "--summary-output", summary, *root_target, *options,
             "--project-root", self.project, self.script],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "stub workload done\n")
        notices = [line for line in process.stderr.splitlines()
                   if line.startswith("metagross:")]
        self.assertEqual(
            notices,
            ["metagross: running target as root"] if root_target else [])
        with open(trace, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        with open(summary, encoding="utf-8") as stream:
            return records, json.load(stream)

    def test_every_stub_call_is_traced_and_attributed(self):
        records, summary = self._trace()
        self.assertEqual(
            [(record["function"], record["api"]) for record in records],
            [("allocate", "cuMemAlloc"), ("upload", "cuMemcpyHtoD"),
             ("launch", "cuLaunchKernel"), ("replay", "cuGraphLaunch"),
             ("wait", "cuCtxSynchronize"),
             ("launch_from_library", "cuLaunchKernelEx"),
             ("copy_around", "cuMemcpyHtoDAsync"),
             ("copy_around", "cuMemcpyDtoH"),
             ("copy_around", "cuMemcpyDtoHAsync"),
             ("copy_around", "cuMemcpyDtoD"),
             ("copy_around", "cuMemcpyDtoDAsync"),
             ("copy_around", "cuMemcpy"), ("copy_around", "cuMemcpyAsync"),
             ("use_pool", "cuMemAllocAsync"), ("use_pool", "cuMemFreeAsync"),
             ("wait_for_stream", "cuStreamSynchronize"),
             ("wait_for_stream", "cuEventSynchronize"),
             ("release", "cuMemFree")])
        by_api = {record["api"]: record for record in records}
        self.assertEqual({record["file"] for record in records}, {self.script})
        self.assertEqual({record["return_code"] for record in records}, {0})
        self.assertEqual(by_api["cuMemAlloc"]["details"]["bytes"], 4096)
        self.assertEqual(by_api["cuMemcpyHtoD"]["details"]["bytes"], 64)
        launch = by_api["cuLaunchKernel"]
        self.assertEqual(launch["kernel"], "stub_kernel")
        self.assertEqual(launch["details"]["grid"], "8,1,1")
        # Block Z, shared memory and stream are the three stack arguments.
        self.assertEqual(launch["details"]["block"], "128,1,2")
        self.assertEqual(launch["details"]["shared"], 48)
        self.assertEqual(launch["details"]["stream"], "0x77")
        launch_ex = by_api["cuLaunchKernelEx"]
        self.assertEqual(launch_ex["kernel"], "library_kernel")
        self.assertEqual(
            {key: launch_ex["details"][key]
             for key in ("grid", "block", "shared", "stream")},
            {"grid": "4,2,1", "block": "64,2,1", "shared": 256,
             "stream": "0x99"})
        self.assertEqual(
            [by_api[api]["details"]["bytes"] for api in (
                "cuMemcpyHtoDAsync", "cuMemcpyDtoH", "cuMemcpyDtoHAsync",
                "cuMemcpyDtoD", "cuMemcpyDtoDAsync", "cuMemcpy",
                "cuMemcpyAsync")],
            [16, 32, 8, 4, 2, 1, 3])
        self.assertEqual(by_api["cuMemFreeAsync"]["details"]["bytes"], 8192)
        self.assertEqual(by_api["cuStreamSynchronize"]["details"],
                         {"stream": "0x77"})
        # Every generated program was loaded: one pair for each API.
        self.assertEqual(summary["configuration"]["attached_probes"],
                         2 * len(_bpf.APIS))
        self.assertEqual(by_api["cuGraphLaunch"]["details"],
                         {"graph_exec": "0xabc0", "stream": "0x77"})
        self.assertEqual(by_api["cuMemFree"]["details"]["bytes"], 4096)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["capture"]["events"], 18)
        self.assertEqual(summary["capture"]["attributed"], 18)
        self.assertEqual(summary["target"]["exit_status"], 0)

    def test_trace_family_selection_limits_the_probes(self):
        records, summary = self._trace("--trace", "memory")
        self.assertEqual([record["api"] for record in records],
                         ["cuMemAlloc", "cuMemAllocAsync", "cuMemFreeAsync",
                          "cuMemFree"])
        self.assertTrue(summary["complete"])

    def test_no_attribution_leaves_functions_unknown(self):
        records, summary = self._trace("--no-attribution")
        self.assertEqual(len(records), 18)
        self.assertEqual({record["function"] for record in records}, {None})
        self.assertTrue(summary["complete"])


if __name__ == "__main__":
    unittest.main()
