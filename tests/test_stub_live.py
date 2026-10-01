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


def allocate():
    pointer = ctypes.c_uint64()
    cuda.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(4096))
    return pointer


def upload(pointer):
    host = ctypes.create_string_buffer(64)
    cuda.cuMemcpyHtoD_v2(pointer, host, ctypes.c_size_t(64))


def launch(function):
    cuda.cuLaunchKernel(function, 8, 1, 1, 128, 1, 1, 0, 0x77, None, None)


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
             ("wait", "cuCtxSynchronize"), ("release", "cuMemFree")])
        by_api = {record["api"]: record for record in records}
        self.assertEqual({record["file"] for record in records}, {self.script})
        self.assertEqual({record["return_code"] for record in records}, {0})
        self.assertEqual(by_api["cuMemAlloc"]["details"]["bytes"], 4096)
        self.assertEqual(by_api["cuMemcpyHtoD"]["details"]["bytes"], 64)
        launch = by_api["cuLaunchKernel"]
        self.assertEqual(launch["kernel"], "stub_kernel")
        self.assertEqual(launch["details"]["grid"], "8,1,1")
        self.assertEqual(launch["details"]["block"], "128,1,1")
        self.assertEqual(launch["details"]["stream"], "0x77")
        self.assertEqual(by_api["cuGraphLaunch"]["details"],
                         {"graph_exec": "0xabc0", "stream": "0x77"})
        self.assertEqual(by_api["cuMemFree"]["details"]["bytes"], 4096)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["capture"]["events"], 6)
        self.assertEqual(summary["capture"]["attributed"], 6)
        self.assertEqual(summary["target"]["exit_status"], 0)

    def test_trace_family_selection_limits_the_probes(self):
        records, summary = self._trace("--trace", "memory")
        self.assertEqual([record["api"] for record in records],
                         ["cuMemAlloc", "cuMemFree"])
        self.assertTrue(summary["complete"])

    def test_no_attribution_leaves_functions_unknown(self):
        records, summary = self._trace("--no-attribution")
        self.assertEqual(len(records), 6)
        self.assertEqual({record["function"] for record in records}, {None})
        self.assertTrue(summary["complete"])


if __name__ == "__main__":
    unittest.main()
