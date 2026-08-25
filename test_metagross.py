# test_metagross.py
"""Tests for metagross. Unprivileged unless RUN_EBPF_INTEGRATION=1."""
import collections
import contextlib
import io
import os
import stat
import tempfile
import unittest

from metagross import (Config, Credentials, MetagrossError, UsageError, exit_status_from_wait,
                       open_trace_output, parse_args, validate_sudo)
from metagross import _bpf
from metagross import _events


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


if __name__ == "__main__":
    unittest.main()
