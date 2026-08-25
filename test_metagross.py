# test_metagross.py
"""Tests for metagross. Unprivileged unless RUN_EBPF_INTEGRATION=1."""
import collections
import os
import stat
import tempfile
import unittest

from metagross import (Config, Credentials, MetagrossError, UsageError, exit_status_from_wait,
                       open_trace_output, parse_args, validate_sudo)
from metagross import _bpf


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


if __name__ == "__main__":
    unittest.main()
