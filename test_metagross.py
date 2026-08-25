# test_metagross.py
"""Tests for metagross. Unprivileged unless RUN_EBPF_INTEGRATION=1."""
import collections
import contextlib
import io
import json
import json as json_mod
import os
import shutil
import stat
import subprocess
import tempfile
import unittest

import metagross
from metagross import (Config, Credentials, MetagrossError, UsageError, exit_status_from_wait,
                       open_trace_output, parse_args, validate_sudo, _validate_script)
from metagross import _bpf
from metagross import _events
from metagross import _profile


INTEGRATION = os.environ.get("RUN_EBPF_INTEGRATION") == "1"


def _raw(api_id, *, args=(), out=0, name=b"", ret=0, ts=0, dur=0, tid=1):
    ev = _bpf.RawEvent(ts=ts, dur=dur, tid=tid, api_id=api_id, ret=ret,
                       out=out, name=name)
    for i, v in enumerate(args):
        ev.args[i] = v
    return ev


class KernelRegistryTest(unittest.TestCase):
    def test_module_get_function_registers(self):
        reg = _events.KernelRegistry()
        api = _bpf.API_BY_ID[18]  # cuModuleGetFunction
        reg.observe(api, _raw(18, out=0xF00, name=b"vec_add"))
        self.assertEqual(reg.name(0xF00), "vec_add")

    def test_kernel_get_function_links_existing_name(self):
        reg = _events.KernelRegistry()
        reg.observe(_bpf.API_BY_ID[19], _raw(19, out=0xAAA, name=b"gemm"))
        reg.observe(_bpf.API_BY_ID[20], _raw(20, args=(0, 0xAAA), out=0xBBB))
        self.assertEqual(reg.name(0xBBB), "gemm")

    def test_failed_registration_ignored(self):
        reg = _events.KernelRegistry()
        reg.observe(_bpf.API_BY_ID[18], _raw(18, out=0xF00, name=b"x", ret=1))
        self.assertIsNone(reg.name(0xF00))


class DescribeTest(unittest.TestCase):
    def setUp(self):
        self.reg = _events.KernelRegistry()
        self.allocs = _events.AllocTracker()

    def test_launch_details(self):
        self.reg.observe(_bpf.API_BY_ID[18], _raw(18, out=0xF00, name=b"vec_add"))
        ev = _raw(1, args=(0xF00, 256, 1, 1, 128, 1, 1, 0, 0x77))
        kernel, det = _events.describe(_bpf.API_BY_ID[1], ev, self.reg, self.allocs)
        self.assertEqual(kernel, "vec_add")
        self.assertEqual(det["grid"], "256,1,1")
        self.assertEqual(det["block"], "128,1,1")
        self.assertEqual(det["stream"], "0x77")

    def test_launch_unknown_kernel_placeholder(self):
        ev = _raw(1, args=(0xDEAD, 1, 1, 1, 1, 1, 1, 0, 0))
        kernel, _ = _events.describe(_bpf.API_BY_ID[1], ev, self.reg, self.allocs)
        self.assertEqual(kernel, "kernel@0xdead")

    def test_alloc_free_tracking(self):
        _, det = _events.describe(_bpf.API_BY_ID[3],
                                  _raw(3, args=(0, 4096), out=0x9000),
                                  self.reg, self.allocs)
        self.assertEqual(det["bytes"], 4096)
        self.assertEqual(self.allocs.total_bytes, 4096)
        _, det = _events.describe(_bpf.API_BY_ID[5], _raw(5, args=(0x9000,)),
                                  self.reg, self.allocs)
        self.assertEqual(det["bytes"], 4096)
        self.assertEqual(self.allocs.total_bytes, 0)

    def test_copy_details(self):
        _, det = _events.describe(_bpf.API_BY_ID[8],
                                  _raw(8, args=(0x1, 0x2, 4194304, 0x77)),
                                  self.reg, self.allocs)
        self.assertEqual(det["bytes"], 4194304)
        self.assertEqual(det["stream"], "0x77")

    def test_shell_quoting(self):
        s = _events.shell_quote_details({"path": "a b", "n": 3})
        self.assertEqual(s, "path='a b' n=3")


class ParseArgsTest(unittest.TestCase):
    def test_minimal(self) -> None:
        cfg: Config = parse_args(["script.py"])
        self.assertEqual(cfg.script, "script.py")
        self.assertEqual(cfg.script_args, [])
        self.assertFalse(cfg.json_output)
        self.assertIsNone(cfg.output_path)
        self.assertEqual(cfg.project_root, ".")

    def test_options_before_script_args_after(self) -> None:
        cfg: Config = parse_args(
            ["--json", "--output", "/tmp/x.jsonl", "--project-root", "/p",
             "script.py", "--json", "positional"])
        self.assertTrue(cfg.json_output)
        self.assertEqual(cfg.output_path, "/tmp/x.jsonl")
        self.assertEqual(cfg.project_root, "/p")
        self.assertEqual(cfg.script_args, ["--json", "positional"])

    def test_ebpf_dump_needs_no_script(self) -> None:
        cfg: Config = parse_args(["--ebpf"])
        self.assertTrue(cfg.dump_ebpf)
        self.assertIsNone(cfg.script)

    def test_missing_script_is_usage_error(self) -> None:
        with self.assertRaises(UsageError):
            parse_args([])

    def test_unknown_option_is_usage_error(self) -> None:
        with self.assertRaises(UsageError):
            parse_args(["--bogus", "script.py"])


FakePw = collections.namedtuple(
    "FakePw", "pw_name pw_uid pw_gid pw_dir")


class ValidateSudoTest(unittest.TestCase):
    def _lookup(self, name):
        if name == "tester":
            return FakePw("tester", 1000, 1000, "/home/tester")
        raise KeyError(name)

    def test_valid_sudo(self):
        env = {"SUDO_UID": "1000", "SUDO_GID": "1000", "SUDO_USER": "tester"}
        creds = validate_sudo(env, self._lookup)
        self.assertEqual(creds, Credentials(1000, 1000, "tester", "/home/tester"))

    def test_direct_root_returns_none(self):
        self.assertIsNone(validate_sudo({}, self._lookup))

    def test_partial_metadata_rejected(self):
        with self.assertRaises(MetagrossError):
            validate_sudo({"SUDO_UID": "1000"}, self._lookup)

    def test_non_numeric_rejected(self):
        env = {"SUDO_UID": "x", "SUDO_GID": "1000", "SUDO_USER": "tester"}
        with self.assertRaises(MetagrossError):
            validate_sudo(env, self._lookup)

    def test_mismatched_uid_rejected(self):
        env = {"SUDO_UID": "1001", "SUDO_GID": "1000", "SUDO_USER": "tester"}
        with self.assertRaises(MetagrossError):
            validate_sudo(env, self._lookup)

    def test_unknown_user_rejected(self):
        env = {"SUDO_UID": "1", "SUDO_GID": "1", "SUDO_USER": "ghost"}
        with self.assertRaises(MetagrossError):
            validate_sudo(env, self._lookup)

    def test_sudo_root_is_direct_root(self):
        env = {"SUDO_UID": "0", "SUDO_GID": "0", "SUDO_USER": "root"}
        self.assertIsNone(validate_sudo(env, lambda n: FakePw("root", 0, 0, "/root")))


class ExitStatusTest(unittest.TestCase):
    def test_exit_codes_preserved(self):
        for code in (0, 1, 42, 255):
            status = os.waitstatus_to_exitcode  # noqa: F841
            self.assertEqual(exit_status_from_wait(code << 8), code)

    def test_signal_death(self):
        self.assertEqual(exit_status_from_wait(2), 130)   # SIGINT
        self.assertEqual(exit_status_from_wait(9), 137)   # SIGKILL


class OutputSafetyTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.uid, self.gid = os.getuid(), os.getgid()

    def _path(self, name):
        return os.path.join(self.dir.name, name)

    def test_new_file_created_0600(self):
        path = self._path("out.jsonl")
        with open_trace_output(path, self.uid, self.gid) as f:
            f.write(b"x")
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)

    def test_existing_owned_regular_file_truncated(self):
        path = self._path("out.jsonl")
        with open(path, "wb") as f:
            f.write(b"old-content")
        with open_trace_output(path, self.uid, self.gid) as f:
            f.write(b"n")
        self.assertEqual(open(path, "rb").read(), b"n")

    def test_symlink_rejected(self):
        target = self._path("real")
        open(target, "wb").close()
        link = self._path("link")
        os.symlink(target, link)
        with self.assertRaises(MetagrossError):
            open_trace_output(link, self.uid, self.gid)

    def test_directory_rejected(self):
        with self.assertRaises(MetagrossError):
            open_trace_output(self.dir.name, self.uid, self.gid)

    def test_foreign_owner_rejected(self):
        path = self._path("owned.jsonl")
        open(path, "wb").close()
        with self.assertRaises(MetagrossError):
            open_trace_output(path, self.uid + 1, self.gid)


class SymbolResolutionTest(unittest.TestCase):
    def test_candidate_symbols(self):
        self.assertEqual(
            _bpf.candidate_symbols("cuMemcpyHtoD"),
            ["cuMemcpyHtoD", "cuMemcpyHtoD_v2", "cuMemcpyHtoD_v3",
             "cuMemcpyHtoD_ptds", "cuMemcpyHtoD_ptsz",
             "cuMemcpyHtoD_v2_ptds", "cuMemcpyHtoD_v2_ptsz"])

    def test_dedupe_by_address(self):
        apis = [_bpf.Api(1, "cuLaunchKernel", "launch", mandatory=True)]
        addrs = {"cuLaunchKernel": 100, "cuLaunchKernel_ptsz": 100}
        got = _bpf.resolve_attachments(apis, addrs.get)
        self.assertEqual([a.symbol for a in got], ["cuLaunchKernel"])

    def test_distinct_addresses_both_attached(self):
        apis = [_bpf.Api(1, "cuLaunchKernel", "launch", mandatory=True)]
        addrs = {"cuLaunchKernel": 100, "cuLaunchKernel_ptsz": 200}
        got = _bpf.resolve_attachments(apis, addrs.get)
        self.assertEqual({a.symbol for a in got},
                         {"cuLaunchKernel", "cuLaunchKernel_ptsz"})

    def test_missing_mandatory_raises(self):
        apis = [_bpf.Api(1, "cuLaunchKernel", "launch", mandatory=True)]
        with self.assertRaises(MetagrossError):
            _bpf.resolve_attachments(apis, lambda s: None)

    def test_missing_optional_skipped(self):
        apis = [_bpf.Api(5, "cuMemAllocAsync", "alloc_async")]
        self.assertEqual(_bpf.resolve_attachments(apis, lambda s: None), [])

    def test_api_table_shape(self):
        ids = [a.api_id for a in _bpf.APIS]
        self.assertEqual(ids, sorted(set(ids)), "api ids must be unique+sorted")
        bases = {a.base for a in _bpf.APIS}
        for required in ("cuLaunchKernel", "cuMemAlloc", "cuMemcpyHtoD",
                         "cuStreamSynchronize", "cuModuleGetFunction"):
            self.assertIn(required, bases)

    def test_dlsym_resolver_against_libc(self):
        import ctypes.util
        resolver = _bpf.dlsym_resolver(ctypes.util.find_library("c"))
        self.assertIsInstance(resolver("read"), int)
        self.assertIsNone(resolver("definitely_not_a_symbol_xyz"))


class BpfSourceTest(unittest.TestCase):
    def setUp(self):
        self.src = _bpf.build_source(4242)

    def test_tgid_substituted(self):
        self.assertIn("4242", self.src)
        self.assertNotIn("TARGET_TGID_PLACEHOLDER", self.src)

    def test_handler_pair_per_api(self):
        for api in _bpf.APIS:
            self.assertIn(f"int enter_{api.base}(", self.src)
            self.assertIn(f"int exit_{api.base}(", self.src)

    def test_braces_balanced(self):
        self.assertEqual(self.src.count("{"), self.src.count("}"))

    def test_launch_reads_stack_args(self):
        self.assertIn("bpf_probe_read_user", self.src)

    def test_decode_event_roundtrip(self):
        raw = _bpf.RawEvent(ts=7, dur=9, tid=5, api_id=1, ret=0)
        raw.args[0] = 0xAB
        raw.name = b"vec_add"
        data = bytes(bytearray(raw))
        ev = _bpf.decode_event(data)
        self.assertEqual((ev.ts, ev.dur, ev.tid, ev.api_id), (7, 9, 5, 1))
        self.assertEqual(ev.args[0], 0xAB)
        self.assertEqual(ev.name, b"vec_add")

    def test_ebpf_flag_prints_source(self):
        import metagross
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = metagross.main(["--ebpf"])
        self.assertEqual(rc, 0)
        self.assertIn("enter_cuLaunchKernel", buf.getvalue())


class RecordCodecTest(unittest.TestCase):
    def test_roundtrip_single(self):
        blob = _profile.encode_record(_profile.CALL, 7, 123456789,
                                      "train_step", "/p/train.py", 31)
        reader = _profile.RecordReader()
        self.assertEqual(reader.feed(blob),
                         [(_profile.CALL, 7, 123456789, "train_step",
                           "/p/train.py", 31)])

    def test_split_feed(self):
        blob = _profile.encode_record(_profile.RETURN, 7, 99, "f", "/p/a.py", 2)
        reader = _profile.RecordReader()
        self.assertEqual(reader.feed(blob[:5]), [])
        self.assertEqual(reader.feed(blob[5:]),
                         [(_profile.RETURN, 7, 99, "f", "/p/a.py", 2)])

    def test_multiple_records_one_feed(self):
        blob = (_profile.encode_record(0, 1, 1, "a", "/p/a.py", 1)
                + _profile.encode_record(1, 1, 2, "a", "/p/a.py", 1))
        self.assertEqual(len(_profile.RecordReader().feed(blob)), 2)


class ProjectFileTest(unittest.TestCase):
    def test_inside_root(self):
        self.assertTrue(_profile.is_project_file("/p/x/y.py", "/p"))

    def test_outside_root(self):
        self.assertFalse(_profile.is_project_file("/usr/lib/python3.10/os.py", "/p"))

    def test_site_packages_inside_root_excluded(self):
        self.assertFalse(
            _profile.is_project_file("/p/venv/lib/python3.10/site-packages/m.py", "/p"))

    def test_metagross_itself_excluded(self):
        import metagross
        path = metagross.__file__
        root = os.path.dirname(os.path.dirname(path))
        self.assertFalse(_profile.is_project_file(path, root))


class InstallHookTest(unittest.TestCase):
    def test_hook_reports_project_calls(self):
        r, w = os.pipe()
        code = (
            "import os, sys, tempfile, textwrap\n"
            "sys.path.insert(0, %r)\n"
            "from metagross import _profile\n"
            "d = tempfile.mkdtemp()\n"
            "p = os.path.join(d, 'proj.py')\n"
            "open(p, 'w').write('def hot():\\n    return 1\\n')\n"
            "sys.path.insert(0, d)\n"
            "_profile.install(%d, d)\n"
            "import proj\n"
            "proj.hot()\n"
            "sys.setprofile(None)\n"
        ) % (os.getcwd(), w)
        pid = os.fork()
        if pid == 0:
            os.close(r)
            exec(code)  # noqa: S102 — test child
            os._exit(0)
        os.close(w)
        data = b""
        while chunk := os.read(r, 4096):
            data += chunk
        os.close(r)
        os.waitpid(pid, 0)
        records = _profile.RecordReader().feed(data)
        funcs = [rec[3] for rec in records]
        self.assertIn("hot", funcs)
        kinds = [rec[0] for rec in records if rec[3] == "hot"]
        self.assertEqual(sorted(set(kinds)), [_profile.CALL, _profile.RETURN])


class AttributionTest(unittest.TestCase):
    def setUp(self):
        self.tl = _events.FrameTimeline()

    def test_deepest_active_frame(self):
        self.tl.on_record(_profile.CALL, 1, 100, "outer", "/p/a.py", 1)
        self.tl.on_record(_profile.CALL, 1, 200, "inner", "/p/a.py", 5)
        info = self.tl.attribute(1, 250)
        self.assertEqual((info.function, info.line), ("inner", 5))

    def test_after_return_attributes_to_caller(self):
        self.tl.on_record(_profile.CALL, 1, 100, "outer", "/p/a.py", 1)
        self.tl.on_record(_profile.CALL, 1, 200, "inner", "/p/a.py", 5)
        self.tl.on_record(_profile.RETURN, 1, 300, "inner", "/p/a.py", 5)
        self.assertEqual(self.tl.attribute(1, 350).function, "outer")

    def test_unknown_before_first_frame(self):
        self.tl.on_record(_profile.CALL, 1, 100, "f", "/p/a.py", 1)
        self.assertIsNone(self.tl.attribute(1, 50))

    def test_unknown_tid(self):
        self.assertIsNone(self.tl.attribute(99, 50))

    def test_orphan_return_ignored(self):
        self.tl.on_record(_profile.RETURN, 1, 100, "ghost", "/p/a.py", 1)
        self.tl.on_record(_profile.CALL, 1, 200, "f", "/p/a.py", 1)
        self.assertEqual(self.tl.attribute(1, 250).function, "f")


class JoinerTest(unittest.TestCase):
    def test_hold_then_release(self):
        j = _events.Joiner(hold_ns=100)
        j.on_profile_record((_profile.CALL, 1, 10, "f", "/p/a.py", 1))
        j.on_gpu_event(_raw(15, args=(0x77,), ts=50, dur=5, tid=1))
        self.assertEqual(j.flush(now_ns=100), [])          # still held
        released = j.flush(now_ns=200)
        self.assertEqual(len(released), 1)
        self.assertEqual(released[0].frame.function, "f")
        self.assertEqual(released[0].api.base, "cuStreamSynchronize")

    def test_late_profile_record_beats_hold(self):
        j = _events.Joiner(hold_ns=100)
        j.on_gpu_event(_raw(16, ts=50, dur=5, tid=1))
        j.on_profile_record((_profile.CALL, 1, 10, "f", "/p/a.py", 1))
        self.assertEqual(j.flush(now_ns=200)[0].frame.function, "f")

    def test_force_flush(self):
        j = _events.Joiner(hold_ns=10**12)
        j.on_gpu_event(_raw(16, ts=50, dur=5, tid=1))
        out = j.flush(now_ns=60, force=True)
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0].frame)

    def test_register_events_feed_registry_not_output(self):
        j = _events.Joiner(hold_ns=0)
        j.on_gpu_event(_raw(18, out=0xF00, name=b"vec_add", ts=1, tid=1))
        self.assertEqual(j.flush(now_ns=10**12), [])
        self.assertEqual(j.registry.name(0xF00), "vec_add")


class RendererTest(unittest.TestCase):
    def _emit(self, ev, json_output):
        j = _events.Joiner()
        j.registry.observe(_bpf.API_BY_ID[18], _raw(18, out=0xF00, name=b"vec_add"))
        buf = io.StringIO()
        r = _events.Renderer(buf, json_output, wall_minus_mono_ns=0,
                             pid=1234, joiner=j)
        r.header()
        r.emit(ev)
        return buf.getvalue()

    def test_table_launch_row(self):
        raw = _raw(1, args=(0xF00, 256, 1, 1, 128, 1, 1, 0, 0x77),
                   ts=3_600_000_000_000, dur=20_000, tid=1)
        ev = _events.AttributedEvent(raw, _bpf.API_BY_ID[1],
                                     _events.FrameInfo("train_step", "/p/train.py", 31))
        out = self._emit(ev, json_output=False)
        self.assertIn("LaunchKernel", out)
        self.assertIn("train_step", out)
        self.assertIn("train.py:31", out)
        self.assertIn("kernel=vec_add", out)
        self.assertIn("0.02ms", out)

    def test_table_unknown_attribution(self):
        raw = _raw(16, ts=1, dur=1, tid=1)
        out = self._emit(_events.AttributedEvent(raw, _bpf.API_BY_ID[16], None),
                         json_output=False)
        self.assertIn("<unknown>", out)

    def test_json_schema(self):
        raw = _raw(1, args=(0xF00, 256, 1, 1, 128, 1, 1, 0, 0x77),
                   ts=1_000_000_000, dur=20_000, tid=5)
        ev = _events.AttributedEvent(raw, _bpf.API_BY_ID[1],
                                     _events.FrameInfo("f", "/p/a.py", 2))
        line = self._emit(ev, json_output=True).strip()
        rec = json.loads(line)
        self.assertEqual(
            sorted(rec),
            ["api", "details", "duration_ns", "file", "function", "kernel",
             "line", "pid", "return_code", "tid", "timestamp"])
        self.assertEqual(rec["api"], "cuLaunchKernel")
        self.assertEqual(rec["kernel"], "vec_add")
        self.assertEqual(rec["pid"], 1234)
        self.assertEqual(rec["tid"], 5)
        self.assertEqual(rec["duration_ns"], 20_000)
        self.assertEqual(rec["details"]["grid"], "256,1,1")

    def test_json_nulls_when_unknown(self):
        raw = _raw(16, ts=1, dur=1, tid=1)
        line = self._emit(_events.AttributedEvent(raw, _bpf.API_BY_ID[16], None),
                          json_output=True).strip()
        rec = json.loads(line)
        self.assertIsNone(rec["function"])
        self.assertIsNone(rec["file"])
        self.assertIsNone(rec["line"])
        self.assertIsNone(rec["kernel"])


class ValidateScriptTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_valid_script(self):
        p = os.path.join(self.dir.name, "s.py")
        open(p, "w").close()
        self.assertEqual(_validate_script(p, self.dir.name), os.path.realpath(p))

    def test_missing_rejected(self):
        with self.assertRaises(MetagrossError):
            _validate_script(os.path.join(self.dir.name, "no.py"), self.dir.name)

    def test_outside_root_rejected(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        p = os.path.join(other.name, "s.py")
        open(p, "w").close()
        with self.assertRaises(MetagrossError):
            _validate_script(p, self.dir.name)

    def test_non_py_rejected(self):
        p = os.path.join(self.dir.name, "s.sh")
        open(p, "w").close()
        with self.assertRaises(MetagrossError):
            _validate_script(p, self.dir.name)

    def test_symlink_out_of_root_rejected(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        target = os.path.join(other.name, "real.py")
        open(target, "w").close()
        link = os.path.join(self.dir.name, "s.py")
        os.symlink(target, link)
        with self.assertRaises(MetagrossError):
            _validate_script(link, self.dir.name)


class MainRoutingTest(unittest.TestCase):
    def test_unprivileged_live_run_fails_cleanly(self):
        if os.geteuid() == 0:
            self.skipTest("running as root")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = metagross.main(["examples/gpu_demo.py"])
        self.assertEqual(rc, 1)
        self.assertIn("root", err.getvalue())


@unittest.skipUnless(INTEGRATION and os.geteuid() == 0,
                     "needs RUN_EBPF_INTEGRATION=1 and root")
class LiveTraceTest(unittest.TestCase):
    def test_cudart_caller_visible(self):
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        if not os.path.exists(nvcc):
            self.skipTest("nvcc unavailable")
        src = "/tmp/metagross_rt.cu"
        binary = "/tmp/metagross_rt"
        with open(src, "w") as f:
            f.write(
                "#include <cuda_runtime.h>\n"
                "__global__ void bump(float *p) { p[threadIdx.x] += 1.0f; }\n"
                "int main() {\n"
                "  float host[32] = {0}; float *dev;\n"
                "  cudaMalloc(&dev, sizeof host);\n"
                "  cudaMemcpy(dev, host, sizeof host, cudaMemcpyHostToDevice);\n"
                "  bump<<<1, 32>>>(dev);\n"
                "  cudaDeviceSynchronize();\n"
                "  cudaMemcpy(host, dev, sizeof host, cudaMemcpyDeviceToHost);\n"
                "  cudaFree(dev);\n"
                "  return host[0] == 1.0f ? 0 : 1;\n"
                "}\n")
        self.addCleanup(os.unlink, src)
        subprocess.run([nvcc, "-o", binary, src], check=True, timeout=300)
        self.addCleanup(os.unlink, binary)
        # A forked child would fall outside the TGID filter, so the wrapper
        # must exec the cudart binary, keeping the traced TGID.
        wrapper = "/tmp/metagross_rt.py"
        with open(wrapper, "w") as f:
            f.write("import os\n"
                    "def run_cudart():\n"
                    f"    os.execv({binary!r}, [{binary!r}])\n"
                    "run_cudart()\n")
        self.addCleanup(os.unlink, wrapper)
        out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        out.close()
        self.addCleanup(os.unlink, out.name)
        proc = subprocess.run(
            ["/usr/bin/python3", "-m", "metagross", "--json", "--output",
             out.name, "--project-root", "/tmp", wrapper],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        records = [json_mod.loads(line) for line in open(out.name) if line.strip()]
        apis = {r["api"] for r in records}
        self.assertTrue(any(a.startswith("cuMemcpyHtoD") for a in apis),
                        f"cudart caller invisible to uprobes; saw {apis}")
        self.assertTrue(any(a.startswith("cuLaunchKernel") for a in apis),
                        f"cudart kernel launch invisible; saw {apis}")

    def _trace_demo(self, *extra):
        out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        out.close()
        self.addCleanup(os.unlink, out.name)
        proc = subprocess.run(
            ["/usr/bin/python3", "-m", "metagross", "--json",
             "--output", out.name, *extra, "examples/gpu_demo.py"],
            capture_output=True, text=True, timeout=120)
        records = [json_mod.loads(line)
                   for line in open(out.name) if line.strip()]
        return proc, records

    def test_end_to_end(self):
        proc, records = self._trace_demo()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("vec_add ok", proc.stdout)      # target stdout untouched
        apis = {r["api"] for r in records}
        for expected in ("cuLaunchKernel", "cuMemcpyHtoD", "cuMemcpyDtoH",
                         "cuStreamSynchronize", "cuMemFree"):
            self.assertTrue(any(a.startswith(expected.rstrip("_")) or a == expected
                                for a in apis), f"missing {expected} in {apis}")
        launches = [r for r in records if r["api"].startswith("cuLaunchKernel")]
        self.assertTrue(launches)
        self.assertEqual(launches[0]["kernel"], "vec_add")
        self.assertEqual(launches[0]["function"], "compute")
        copies = [r for r in records if r["api"].startswith("cuMemcpyHtoD")]
        self.assertEqual(copies[0]["details"]["bytes"], 4096)
        self.assertEqual(copies[0]["function"], "upload")
        # The driver may route internal allocations through its own public
        # exports, so assert on the demo's three 4096-byte allocs, not totals.
        demo_allocs = [r for r in records
                       if r["api"].startswith("cuMemAlloc")
                       and r["details"].get("bytes") == 4096
                       and r["function"] == "upload"]
        self.assertEqual(len(demo_allocs), 3)
        frees = [r for r in records if r["api"].startswith("cuMemFree")]
        self.assertGreaterEqual(len(frees), 3)

    def test_exit_status_forwarded(self):
        proc = subprocess.run(
            ["/usr/bin/python3", "-m", "metagross", "--project-root", "/tmp",
             self._failing_script()], capture_output=True, timeout=120)
        self.assertEqual(proc.returncode, 42)

    def _failing_script(self):
        path = "/tmp/metagross_exit42.py"
        with open(path, "w") as f:
            f.write("import sys\nsys.exit(42)\n")
        self.addCleanup(os.unlink, path)
        return path

    def test_target_runs_as_invoker(self):
        if "SUDO_UID" not in os.environ:
            self.skipTest("not under sudo")
        path = "/tmp/metagross_whoami.py"
        with open(path, "w") as f:
            f.write("import os\nprint('uid', os.getuid())\n")
        self.addCleanup(os.unlink, path)
        proc = subprocess.run(
            ["/usr/bin/python3", "-m", "metagross", "--project-root", "/tmp",
             path], capture_output=True, text=True, timeout=120)
        self.assertIn(f"uid {os.environ['SUDO_UID']}", proc.stdout)

    def test_ctrl_c_returns_130(self):
        path = "/tmp/metagross_sleep.py"
        with open(path, "w") as f:
            f.write("import time\nprint('sleeping', flush=True)\ntime.sleep(60)\n")
        self.addCleanup(os.unlink, path)
        proc = subprocess.Popen(
            ["/usr/bin/python3", "-m", "metagross", "--project-root", "/tmp",
             path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertIn("sleeping", proc.stdout.readline())
        proc.send_signal(2)  # SIGINT, as Ctrl-C would deliver
        proc.wait(timeout=60)
        self.assertEqual(proc.returncode, 130)

    def test_no_bpf_programs_left_behind(self):
        if not shutil.which("bpftool"):
            self.skipTest("bpftool unavailable")
        before = subprocess.run(["bpftool", "prog", "list"],
                                capture_output=True, text=True).stdout
        self._trace_demo()
        after = subprocess.run(["bpftool", "prog", "list"],
                               capture_output=True, text=True).stdout
        self.assertLessEqual(len(after.splitlines()), len(before.splitlines()) + 1)


if __name__ == "__main__":
    unittest.main()
