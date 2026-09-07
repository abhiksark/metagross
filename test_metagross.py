# test_metagross.py
"""Tests for metagross. Unprivileged unless RUN_EBPF_INTEGRATION=1."""
import collections
import contextlib
import io
import json
import http.server
import os
import runpy
import shutil
import stat
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import threading
import time

import metagross
from metagross import (Config, Credentials, MetagrossError, UsageError,
                       _validate_output_paths, _validate_script,
                       exit_status_from_wait, open_trace_output, parse_args,
                       validate_sudo)
from metagross import _bpf
from metagross import _events
from metagross import _profile
from metagross import _publish
from metagross import _web


INTEGRATION = os.environ.get("RUN_EBPF_INTEGRATION") == "1"


def _fresh_output_path(test):
    """Return a not-yet-created trace-output path metagross can create and own.

    The live suite runs as root; pre-creating the file would make it
    root-owned and metagross (dropped to the invoking uid) would rightly
    refuse to truncate it. Letting metagross create the file matches real
    usage and lets it chown the file to the invoker.
    """
    directory = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, directory, ignore_errors=True)
    return os.path.join(directory, "trace.jsonl")


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

    def test_free_failure_does_not_skew_gpu_total(self):
        _, det = _events.describe(_bpf.API_BY_ID[3],
                                  _raw(3, args=(0, 4096), out=0x9000),
                                  self.reg, self.allocs)
        self.assertEqual(self.allocs.total_bytes, 4096)
        # A failed cuMemFree (ret != 0) must not remove the allocation from
        # tracking or decrement gpu_total; the ptr detail still renders.
        _, det = _events.describe(_bpf.API_BY_ID[5],
                                  _raw(5, args=(0x9000,), ret=1),
                                  self.reg, self.allocs)
        self.assertNotIn("bytes", det)
        self.assertEqual(det["ptr"], "0x9000")
        self.assertEqual(self.allocs.total_bytes, 4096)

    def test_launch_masks_dirty_high_bits(self):
        # A dirty high half of a 64-bit register in a 32-bit dimension
        # argument must not corrupt the printed grid/block dims.
        dirty = 0xDEADBEEF00000010
        ev = _raw(1, args=(0xF00, dirty, dirty, dirty, dirty, dirty, dirty,
                           0, 0x77))
        _, det = _events.describe(_bpf.API_BY_ID[1], ev, self.reg, self.allocs)
        self.assertEqual(det["grid"], "16,16,16")
        self.assertEqual(det["block"], "16,16,16")

    def test_copy_details(self):
        _, det = _events.describe(_bpf.API_BY_ID[8],
                                  _raw(8, args=(0x1, 0x2, 4194304, 0x77)),
                                  self.reg, self.allocs)
        self.assertEqual(det["bytes"], 4194304)
        self.assertEqual(det["stream"], "0x77")

    def test_shell_quoting(self):
        s = _events.shell_quote_details({"path": "a b", "n": 3})
        self.assertEqual(s, "path='a b' n=3")


class CaptureStatsTest(unittest.TestCase):
    def test_snapshot_aggregates_capture_memory_copy_and_timing(self):
        joiner = _events.Joiner()
        stats = _events.CaptureStats()
        frame = _events.FrameInfo("step", "/p/train.py", 10)
        attributed = [
            _events.AttributedEvent(
                _raw(3, args=(0, 1024), out=0xA, dur=10),
                _bpf.API_BY_ID[3], frame,
            ),
            _events.AttributedEvent(
                _raw(7, args=(0xA, 0xB, 512), dur=20),
                _bpf.API_BY_ID[7], frame,
            ),
            _events.AttributedEvent(
                _raw(9, args=(0xB, 0xA, 256), ret=1, dur=30),
                _bpf.API_BY_ID[9], frame,
            ),
            _events.AttributedEvent(
                _raw(15, args=(0x77,), dur=40),
                _bpf.API_BY_ID[15], None,
            ),
        ]
        for event in attributed:
            stats.observe(joiner.enrich(event))

        snapshot = stats.snapshot(
            lost_events=0,
            dropped_nested_calls=0,
            observed_outstanding_bytes=joiner.allocs.total_bytes,
            render_failed=False,
        )
        self.assertTrue(snapshot["complete"])
        self.assertEqual(snapshot["capture"]["events"], 4)
        self.assertEqual(snapshot["capture"]["attributed"], 3)
        self.assertEqual(snapshot["capture"]["unknown_attribution"], 1)
        self.assertEqual(snapshot["capture"]["cuda_errors"], 1)
        self.assertEqual(snapshot["timing"]["total_api_duration_ns"], 100)
        self.assertEqual(snapshot["timing"]["synchronization_duration_ns"], 40)
        self.assertEqual(snapshot["memory"]["successful_allocation_bytes"], 1024)
        self.assertEqual(snapshot["memory"]["observed_peak_bytes"], 1024)
        self.assertEqual(snapshot["memory"]["observed_outstanding_bytes"], 1024)
        self.assertEqual(
            snapshot["copies"]["successful_bytes_by_api"],
            {"cuMemcpyHtoD": 512},
        )
        self.assertEqual(snapshot["top_functions"][0]["function"], "step")
        self.assertEqual(snapshot["top_functions"][0]["count"], 3)

    def test_loss_or_render_failure_marks_snapshot_incomplete(self):
        stats = _events.CaptureStats()
        snapshot = stats.snapshot(
            lost_events=2,
            dropped_nested_calls=0,
            observed_outstanding_bytes=0,
            render_failed=False,
        )
        self.assertFalse(snapshot["complete"])
        snapshot = stats.snapshot(
            lost_events=0,
            dropped_nested_calls=0,
            observed_outstanding_bytes=0,
            render_failed=True,
        )
        self.assertFalse(snapshot["complete"])


class ParseArgsTest(unittest.TestCase):
    def test_minimal(self) -> None:
        cfg: Config = parse_args(["script.py"])
        self.assertEqual(cfg.script, "script.py")
        self.assertEqual(cfg.script_args, [])
        self.assertFalse(cfg.json_output)
        self.assertIsNone(cfg.output_path)
        self.assertEqual(cfg.project_root, ".")
        self.assertFalse(cfg.show_stats)
        self.assertIsNone(cfg.summary_output_path)
        self.assertTrue(cfg.python_attribution)
        self.assertIsNone(cfg.trace_families)

    def test_options_before_script_args_after(self) -> None:
        cfg: Config = parse_args(
            ["--json", "--output", "/tmp/x.jsonl", "--project-root", "/p",
             "script.py", "--json", "positional"])
        self.assertTrue(cfg.json_output)
        self.assertEqual(cfg.output_path, "/tmp/x.jsonl")
        self.assertEqual(cfg.project_root, "/p")
        self.assertEqual(cfg.script_args, ["--json", "positional"])

    def test_capture_policy_options(self) -> None:
        cfg = parse_args([
            "--trace", "launch,sync", "--no-attribution", "--stats",
            "--summary-output", "/tmp/summary.json", "script.py",
        ])
        self.assertEqual(cfg.trace_families, frozenset(("launch", "sync")))
        self.assertFalse(cfg.python_attribution)
        self.assertTrue(cfg.show_stats)
        self.assertEqual(cfg.summary_output_path, "/tmp/summary.json")

    def test_dashboard_port_and_target_argument_boundary(self) -> None:
        cfg = parse_args(
            ["--dashboard-port", "8765", "script.py", "--dashboard-port", "7"]
        )
        self.assertEqual(cfg.dashboard_port, 8765)
        self.assertEqual(cfg.script_args, ["--dashboard-port", "7"])

    def test_dashboard_port_rejects_bounds_and_ebpf(self) -> None:
        for value in ("0", "65536", "not-a-port"):
            with self.subTest(value=value), self.assertRaises(UsageError):
                parse_args(["--dashboard-port", value, "script.py"])
        for argv in (
            ["--ebpf", "--dashboard-port", "8765"],
            ["--dashboard-port", "8765", "--ebpf"],
        ):
            with self.subTest(argv=argv), self.assertRaisesRegex(
                UsageError, "cannot be combined"
            ):
                parse_args(argv)

    def test_trace_all_uses_default_selection(self) -> None:
        cfg = parse_args(["--trace", "all", "script.py"])
        self.assertIsNone(cfg.trace_families)

    def test_invalid_trace_family_is_usage_error(self) -> None:
        with self.assertRaisesRegex(UsageError, "unknown trace family"):
            parse_args(["--trace", "launch,banana", "script.py"])
        with self.assertRaisesRegex(UsageError, "cannot be combined"):
            parse_args(["--trace", "all,sync", "script.py"])
        with self.assertRaisesRegex(UsageError, "comma-separated"):
            parse_args(["--trace", ",", "script.py"])

    def test_value_options_require_values(self) -> None:
        for option in ("--trace", "--summary-output", "--dashboard-port"):
            with self.subTest(option=option):
                with self.assertRaisesRegex(UsageError, "requires a value"):
                    parse_args([option])

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


class BenchmarkHarnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = runpy.run_path(
            "examples/docker/benchmark_overhead.py"
        )

    def test_bare_command_bypasses_metagross(self):
        command = self.benchmark["_docker_command"](
            "image:test", "basic_tensor_ops.py", "bare"
        )
        self.assertIn("--entrypoint", command)
        self.assertIn("/usr/bin/python3", command)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("--no-attribution", command)

    def test_launch_only_command_uses_capture_controls(self):
        command = self.benchmark["_docker_command"](
            "image:test", "basic_tensor_ops.py", "launch-only"
        )
        self.assertIn("--privileged", command)
        self.assertIn("--pid=host", command)
        self.assertIn("--no-attribution", command)
        trace_index = command.index("--trace")
        self.assertEqual(command[trace_index + 1], "launch")

    def test_steady_state_command_uses_benchmark_target(self):
        command = self.benchmark["_docker_command"](
            "image:test", "rapid-launches", "full"
        )
        self.assertIn("/workspace/benchmarks/steady_state.py", command)
        self.assertIn("--iterations", command)
        self.assertIn("2000", command)

    def test_summary_counts_loss_and_uses_median(self):
        summary = self.benchmark["_summarize"]([
            {"elapsed_seconds": 3.0, "lost_events": 2,
             "dropped_nested_calls": 0},
            {"elapsed_seconds": 1.0, "lost_events": 0,
             "dropped_nested_calls": 1},
            {"elapsed_seconds": 2.0, "lost_events": 3,
             "dropped_nested_calls": 0},
        ])
        self.assertEqual(summary["median_seconds"], 2.0)
        self.assertEqual(summary["lost_events"], 5)
        self.assertEqual(summary["dropped_nested_calls"], 1)
        self.assertIsNone(summary["target_median_seconds"])


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
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"n")

    def test_invalid_final_path_components_preserve_file(self):
        path = self._path("trace")
        for suffix in ("/", "/.", "/.."):
            with self.subTest(suffix=suffix):
                with open(path, "wb") as handle:
                    handle.write(b"keep this capture")
                with self.assertRaises(MetagrossError):
                    with open_trace_output(path + suffix, self.uid, self.gid):
                        pass
                with open(path, "rb") as handle:
                    self.assertEqual(handle.read(), b"keep this capture")

    def test_symlink_parent_before_dot_dot_preserves_both_destinations(self):
        directory, elsewhere = self._path("directory"), self._path("elsewhere")
        for parent in (directory, elsewhere, os.path.join(elsewhere, "child")):
            os.mkdir(parent, 0o700)
        link = os.path.join(directory, "link")
        os.symlink(os.path.join(elsewhere, "child"), link)
        destinations = (os.path.join(directory, "trace"),
                        os.path.join(elsewhere, "trace"))
        for destination in destinations:
            with open(destination, "wb") as handle:
                handle.write(b"keep this capture")
        path = os.path.join(link, "..", "trace")
        with self.assertRaises(MetagrossError):
            with open_trace_output(path, self.uid, self.gid):
                pass
        for destination in destinations:
            with open(destination, "rb") as handle:
                self.assertEqual(handle.read(), b"keep this capture")

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

    def test_trace_and_summary_outputs_must_be_distinct(self):
        trace = self._path("trace.jsonl")
        with self.assertRaisesRegex(MetagrossError, "must be different"):
            _validate_output_paths(trace, os.path.join(self.dir.name, ".", "trace.jsonl"))
        _validate_output_paths(trace, self._path("summary.json"))

    def test_replaced_inode_is_validated_before_truncation(self):
        path = self._path("out.jsonl")
        victim = self._path("protected")
        with open(path, "wb") as handle:
            handle.write(b"old trace")
        with open(victim, "wb") as handle:
            handle.write(b"protected content")
        victim_inode = os.stat(victim).st_ino
        real_open, real_fstat = os.open, os.fstat
        writable_opens = []

        def replace_before_open(name, flags, *args, **kwargs):
            if flags & (os.O_PATH | os.O_WRONLY) and not flags & os.O_CREAT:
                os.replace(victim, path)
            fd = real_open(name, flags, *args, **kwargs)
            if flags & os.O_WRONLY:
                writable_opens.append(name)
            return fd

        def foreign_owner(fd):
            info = real_fstat(fd)
            if info.st_ino == victim_inode:
                values = list(info)
                values[4] = self.uid + 1
                return os.stat_result(values)
            return info

        # Only ownership needs a stand-in in this unprivileged test. The
        # pathname replacement, opened inode, and file contents are real.
        with mock.patch("metagross.os.open", side_effect=replace_before_open), \
                mock.patch("metagross.os.fstat", side_effect=foreign_owner):
            with self.assertRaisesRegex(MetagrossError, "not owned by uid"):
                with open_trace_output(path, self.uid, self.gid):
                    pass
        self.assertEqual(writable_opens, [])
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), b"protected content")

    def test_character_device_is_rejected_before_any_writable_open(self):
        real_open = os.open
        writable_opens = []

        def record_open(name, flags, *args, **kwargs):
            fd = real_open(name, flags, *args, **kwargs)
            if flags & os.O_ACCMODE in (os.O_WRONLY, os.O_RDWR):
                writable_opens.append(name)
            return fd

        with mock.patch("metagross.os.open", side_effect=record_open):
            with self.assertRaisesRegex(MetagrossError, "not a regular file"):
                with open_trace_output("/dev/null", self.uid, self.gid):
                    pass
        self.assertEqual(writable_opens, [])

    def test_existing_file_reopen_failure_closes_descriptors(self):
        path = self._path("trace")
        with open(path, "wb") as handle:
            handle.write(b"keep this capture")
        real_open = os.open
        descriptors = []

        def fail_reopen(name, flags, *args, **kwargs):
            if str(name).startswith("/proc/self/fd/"):
                raise OSError("injected reopen failure")
            fd = real_open(name, flags, *args, **kwargs)
            descriptors.append(fd)
            return fd

        with mock.patch("metagross.os.open", side_effect=fail_reopen):
            with self.assertRaisesRegex(MetagrossError, "injected reopen failure"):
                with open_trace_output(path, self.uid, self.gid):
                    pass
        for fd in descriptors:
            with self.assertRaises(OSError):
                os.fstat(fd)
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), b"keep this capture")

    def test_existing_file_reopen_keeps_the_pinned_inode(self):
        path, moved = self._path("trace"), self._path("moved")
        victim = self._path("protected")
        with open(path, "wb") as handle:
            handle.write(b"old trace")
        with open(victim, "wb") as handle:
            handle.write(b"protected content")
        real_open = os.open

        def replace_before_reopen(name, flags, *args, **kwargs):
            if flags & os.O_WRONLY and not flags & os.O_CREAT:
                os.rename(path, moved)
                os.replace(victim, path)
            return real_open(name, flags, *args, **kwargs)

        with mock.patch("metagross.os.open", side_effect=replace_before_reopen):
            with open_trace_output(path, self.uid, self.gid) as handle:
                handle.write(b"new trace")
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), b"protected content")
        with open(moved, "rb") as handle:
            self.assertEqual(handle.read(), b"new trace")

    def test_output_failures_close_opened_descriptors(self):
        real_open, real_fstat = os.open, os.fstat
        for operation in ("fstat", "fchown", "ftruncate", "fdopen"):
            with self.subTest(operation=operation):
                path = self._path(operation)
                descriptors = []

                def record_open(*args, **kwargs):
                    fd = real_open(*args, **kwargs)
                    descriptors.append(fd)
                    return fd

                def fail_on_file(fd, *args, **kwargs):
                    if stat.S_ISDIR(real_fstat(fd).st_mode):
                        return real_fstat(fd)
                    raise OSError("injected output failure")

                with mock.patch("metagross.os.open", side_effect=record_open), \
                        mock.patch("metagross.os." + operation,
                                   side_effect=fail_on_file):
                    with self.assertRaises(MetagrossError):
                        with open_trace_output(path, self.uid, self.gid):
                            pass
                for fd in descriptors:
                    with self.assertRaises(OSError):
                        real_fstat(fd)

    def test_unsafe_parent_directories_are_rejected(self):
        directory = self._path("directory")
        os.mkdir(directory)
        alias = self._path("alias")
        os.symlink(directory, alias)
        with self.assertRaises(MetagrossError):
            with open_trace_output(os.path.join(alias, "trace"),
                                   self.uid, self.gid):
                pass
        self.assertFalse(os.path.exists(os.path.join(directory, "trace")))
        os.chmod(directory, 0o777)
        with self.assertRaises(MetagrossError):
            with open_trace_output(os.path.join(directory, "trace"),
                                   self.uid, self.gid):
                pass
        self.assertFalse(os.path.exists(os.path.join(directory, "trace")))

    def test_owned_sticky_parent_is_supported(self):
        directory = self._path("sticky")
        os.mkdir(directory)
        os.chmod(directory, 0o1777)
        path = os.path.join(directory, "trace")
        with open_trace_output(path, self.uid, self.gid) as handle:
            handle.write(b"trace")
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), b"trace")

    def test_fifo_is_rejected_without_waiting_for_a_reader(self):
        path = self._path("fifo")
        os.mkfifo(path)
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import sys, metagross\n"
             "try:\n"
             "    metagross.open_trace_output(sys.argv[1], int(sys.argv[2]), "
             "int(sys.argv[3]))\n"
             "except metagross.MetagrossError as exc:\n"
             "    assert 'not a regular file' in str(exc), str(exc)\n"
             "    raise SystemExit(0)\n"
             "raise SystemExit(1)\n", path, str(self.uid), str(self.gid)],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_empty_output_paths_keep_the_default_streams(self):
        for trace_name, summary_name in (("", None), (None, ""),
                                         ("", "summary"), ("trace", "")):
            with self.subTest(trace=trace_name, summary=summary_name):
                trace = self._path(trace_name) if trace_name else trace_name
                summary = self._path(summary_name) if summary_name else summary_name
                with contextlib.ExitStack() as opened:
                    stderr = opened.enter_context(
                        contextlib.redirect_stderr(io.StringIO())
                    )
                    try:
                        trace_stream, summary_stream = metagross._open_output_streams(
                            trace, summary, self.uid, self.gid
                        )
                    except MetagrossError as exc:
                        self.fail(f"empty output path must use the default: {exc}")
                    if trace:
                        opened.enter_context(trace_stream).write("trace")
                    else:
                        self.assertIs(trace_stream, stderr)
                    if summary:
                        opened.enter_context(summary_stream).write("summary")
                    else:
                        self.assertIsNone(summary_stream)
                for path, expected in ((trace, "trace"), (summary, "summary")):
                    if path:
                        with open(path, encoding="utf-8") as handle:
                            self.assertEqual(handle.read(), expected)

    def _run_with_outputs(self, trace, summary, expected_error):
        cfg = Config(script=__file__, project_root=os.path.dirname(__file__),
                     output_path=trace, summary_output_path=summary)
        creds = Credentials(self.uid, self.gid, "fixture", self.dir.name)
        with mock.patch("metagross.os.geteuid", return_value=0), \
                mock.patch("metagross.validate_sudo", return_value=creds), \
                mock.patch("metagross._bpf.find_libcuda", return_value="/unused"), \
                mock.patch.dict("sys.modules", {"bcc": mock.Mock(BPF=object)}), \
                mock.patch("metagross.os.fork",
                           side_effect=AssertionError("target must not start")):
            with self.assertRaisesRegex(MetagrossError, expected_error):
                metagross.run_live(cfg)

    def test_hard_linked_outputs_are_rejected_without_truncation(self):
        trace, summary = self._path("trace"), self._path("summary")
        with open(trace, "wb") as handle:
            handle.write(b"keep this capture")
        os.link(trace, summary)
        self._run_with_outputs(trace, summary, "must be different files")
        with open(trace, "rb") as handle:
            self.assertEqual(handle.read(), b"keep this capture")

    def test_output_alias_introduced_during_open_is_rejected(self):
        trace, summary = self._path("trace"), self._path("summary")
        with open(trace, "wb") as handle:
            handle.write(b"keep this capture")
        real_open_output = open_trace_output

        def alias_summary(path, *args, **kwargs):
            if path == summary:
                os.link(trace, summary)
            return real_open_output(path, *args, **kwargs)

        with mock.patch("metagross.open_trace_output", side_effect=alias_summary):
            self._run_with_outputs(trace, summary, "must be different files")
        with open(trace, "rb") as handle:
            self.assertEqual(handle.read(), b"keep this capture")

    def test_invalid_summary_preserves_existing_trace(self):
        trace = self._path("trace")
        with open(trace, "wb") as handle:
            handle.write(b"keep this capture")
        self._run_with_outputs(trace, self.dir.name,
                               "not a regular file|Is a directory")
        with open(trace, "rb") as handle:
            self.assertEqual(handle.read(), b"keep this capture")

    @unittest.skipUnless(os.geteuid() == 0, "needs root for real output ownership")
    def test_real_foreign_inode_swap_preserves_contents(self):
        trace, victim = self._path("trace"), self._path("protected")
        with open(trace, "wb") as handle:
            handle.write(b"old trace")
        os.chown(trace, 1000, 1000)
        with open(victim, "wb") as handle:
            handle.write(b"protected content")
        os.chmod(victim, 0o600)
        real_open = os.open

        def replace_before_open(name, flags, *args, **kwargs):
            if flags & (os.O_PATH | os.O_WRONLY) and not flags & os.O_CREAT:
                os.replace(victim, trace)
            return real_open(name, flags, *args, **kwargs)

        with mock.patch("metagross.os.open", side_effect=replace_before_open):
            with self.assertRaises(MetagrossError):
                with open_trace_output(trace, 1000, 1000):
                    pass
        self.assertEqual(os.stat(trace).st_uid, 0)
        with open(trace, "rb") as handle:
            self.assertEqual(handle.read(), b"protected content")

    @unittest.skipUnless(os.geteuid() == 0, "needs root for real output ownership")
    def test_new_file_chown_cannot_follow_a_replacement_symlink(self):
        trace, moved = self._path("trace"), self._path("moved")
        victim = self._path("protected")
        with open(victim, "wb") as handle:
            handle.write(b"protected content")
        os.chmod(victim, 0o600)
        real_open = os.open

        def replace_after_create(name, flags, *args, **kwargs):
            fd = real_open(name, flags, *args, **kwargs)
            if flags & os.O_CREAT:
                os.rename(trace, moved)
                os.symlink(victim, trace)
            return fd

        with mock.patch("metagross.os.open", side_effect=replace_after_create):
            with open_trace_output(trace, 1000, 1000) as handle:
                handle.write(b"new trace")
        self.assertEqual((os.stat(victim).st_uid, os.stat(victim).st_gid), (0, 0))
        self.assertEqual((os.stat(moved).st_uid, os.stat(moved).st_gid), (1000, 1000))
        self.assertEqual(stat.S_IMODE(os.stat(moved).st_mode), 0o600)
        with open(victim, "rb") as handle:
            self.assertEqual(handle.read(), b"protected content")
        with open(moved, "rb") as handle:
            self.assertEqual(handle.read(), b"new trace")

    @unittest.skipUnless(os.geteuid() == 0, "needs root for real output ownership")
    def test_foreign_owned_parent_is_rejected(self):
        directory = self._path("foreign")
        os.mkdir(directory)
        os.chown(directory, 2000, 2000)
        path = os.path.join(directory, "trace")
        with self.assertRaises(MetagrossError):
            with open_trace_output(path, 1000, 1000):
                pass
        self.assertFalse(os.path.exists(path))


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

    def test_select_launch_apis_includes_registration_dependency(self):
        selected = _bpf.select_apis(frozenset(("launch",)))
        categories = {api.category for api in selected}
        self.assertEqual(categories, {"launch", "launch_ex", "register"})

    def test_select_multiple_api_families(self):
        selected = _bpf.select_apis(frozenset(("copy", "sync")))
        categories = {api.category for api in selected}
        self.assertIn("copy_h2d", categories)
        self.assertIn("copy_generic", categories)
        self.assertEqual(categories - {"copy_h2d", "copy_d2h", "copy_d2d",
                                       "copy_generic", "sync"}, set())

    def test_select_all_apis_returns_copy(self):
        selected = _bpf.select_apis()
        self.assertEqual(selected, _bpf.APIS)
        self.assertIsNot(selected, _bpf.APIS)

    def test_dlsym_resolver_against_libc(self):
        import ctypes.util
        resolver = _bpf.dlsym_resolver(ctypes.util.find_library("c"))
        self.assertIsInstance(resolver("read"), int)
        self.assertIsNone(resolver("definitely_not_a_symbol_xyz"))

    def test_find_libcuda_returns_mocked_existing_absolute_path(self):
        expected = _bpf._LIBCUDA_CANDIDATES[1]
        with (mock.patch.object(_bpf.ctypes.util, "find_library",
                                return_value=None),
              mock.patch.object(_bpf.os.path, "exists",
                                side_effect=lambda path: path == expected),
              mock.patch.object(_bpf, "_ldconfig_libcuda_path",
                                return_value=None)):
            path = _bpf.find_libcuda()
        self.assertEqual(path, expected)
        self.assertTrue(path.startswith("/"), f"not absolute: {path!r}")


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

    def test_ring_buffer_has_four_mibibyte_capacity(self):
        self.assertIn("BPF_RINGBUF_OUTPUT(events, 1024)", self.src)

    def test_filtered_source_contains_only_selected_families(self):
        selected = _bpf.select_apis(frozenset(("sync",)))
        src = _bpf.build_source(4242, selected)
        self.assertIn("enter_cuStreamSynchronize", src)
        self.assertNotIn("enter_cuLaunchKernel", src)
        self.assertNotIn("enter_cuMemAlloc", src)

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

    def test_ebpf_flag_honors_trace_selection(self):
        import metagross
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = metagross.main(["--trace", "sync", "--ebpf"])
        self.assertEqual(rc, 0)
        self.assertIn("enter_cuStreamSynchronize", buf.getvalue())
        self.assertNotIn("enter_cuLaunchKernel", buf.getvalue())


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

    def test_truncated_multibyte_does_not_raise(self):
        # "x" + "e"-acute * 300 is 601 bytes; the raw 500-byte truncation in
        # encode_record cuts the 500th byte inside a two-byte "e"-acute
        # character (byte 499 is its leading 0xC3), so a strict utf-8
        # decode of the truncated bytes raises UnicodeDecodeError. feed()
        # must not raise; it replaces the partial character and still
        # yields exactly one record.
        long_str = "x" + "é" * 300
        blob = _profile.encode_record(_profile.CALL, 3, 42, long_str,
                                      long_str, 5)
        records = _profile.RecordReader().feed(blob)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0][3].endswith("�"))
        self.assertTrue(records[0][4].endswith("�"))


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

    def test_filesystem_root_accepts_project_file(self):
        self.assertTrue(_profile.is_project_file("/tmp/project.py", "/"))

    def test_classifier_caches_path_resolution(self):
        realpath = _profile.os.path.realpath
        with mock.patch.object(_profile.os.path, "realpath",
                               wraps=realpath) as resolve:
            classifier = _profile._ProjectClassifier("/p")
            self.assertTrue(classifier.includes("/p/project.py"))
            self.assertTrue(classifier.includes("/p/project.py"))
        self.assertEqual(resolve.call_count, 2)  # root once, project path once


class InstallHookTest(unittest.TestCase):
    def test_hook_reports_project_calls(self):
        r, w = os.pipe()
        code = (
            "import os, sys, tempfile, textwrap\n"
            "sys.path.insert(0, %r)\n"
            "from metagross import _profile\n"
            "d = tempfile.mkdtemp()\n"
            "p = os.path.join(d, 'proj.py')\n"
            "with open(p, 'w') as stream:\n"
            "    stream.write('def hot():\\n    return 1\\n')\n"
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

    def test_monotonic_queries_advance_incremental_cursor(self):
        self.tl.on_record(_profile.CALL, 1, 100, "outer", "/p/a.py", 1)
        self.tl.on_record(_profile.CALL, 1, 200, "inner", "/p/a.py", 2)
        self.tl.on_record(_profile.RETURN, 1, 300, "inner", "/p/a.py", 2)
        self.assertEqual(self.tl.attribute(1, 250).function, "inner")
        self.assertEqual(self.tl._states[1].index, 2)
        self.assertEqual(self.tl.attribute(1, 350).function, "outer")
        self.assertEqual(self.tl._states[1].index, 3)

    def test_late_profile_record_rebuilds_incremental_stack(self):
        self.tl.on_record(_profile.CALL, 1, 100, "outer", "/p/a.py", 1)
        self.assertEqual(self.tl.attribute(1, 200).function, "outer")
        self.tl.on_record(_profile.CALL, 1, 150, "inner", "/p/a.py", 2)
        self.assertEqual(self.tl.attribute(1, 250).function, "inner")

    def test_older_query_does_not_rewind_monotonic_cursor(self):
        self.tl.on_record(_profile.CALL, 1, 100, "outer", "/p/a.py", 1)
        self.tl.on_record(_profile.RETURN, 1, 300, "outer", "/p/a.py", 1)
        self.assertIsNone(self.tl.attribute(1, 350))
        self.assertEqual(self.tl.attribute(1, 150).function, "outer")
        self.assertIsNone(self.tl.attribute(1, 400))

    def test_prune_preserves_incremental_active_stack(self):
        self.tl.on_record(_profile.CALL, 1, 100, "done", "/p/a.py", 1)
        self.tl.on_record(_profile.RETURN, 1, 150, "done", "/p/a.py", 1)
        self.tl.on_record(_profile.CALL, 1, 200, "active", "/p/a.py", 2)
        self.assertEqual(self.tl.attribute(1, 250).function, "active")
        self.tl.prune(175)
        self.assertEqual(len(self.tl._logs[1]), 1)
        self.assertEqual(self.tl.attribute(1, 300).function, "active")

    def test_prune_resets_stack_when_cut_includes_unreplayed_return(self):
        self.tl.on_record(_profile.CALL, 1, 100, "done", "/p/a.py", 1)
        self.tl.on_record(_profile.RETURN, 1, 150, "done", "/p/a.py", 1)
        self.assertEqual(self.tl.attribute(1, 120).function, "done")
        self.tl.prune(175)
        self.assertIsNone(self.tl.attribute(1, 200))


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
        r = _events.Renderer(
            buf, json_output, wall_minus_mono_ns=0, pid=1234
        )
        r.header()
        r.emit(j.enrich(ev))
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
        joiner = _events.Joiner()
        joiner.registry.observe(
            _bpf.API_BY_ID[18],
            _raw(18, out=0xF00, name=b"vec_add"),
        )
        canonical = _events.event_record(
            joiner.enrich(ev),
            wall_minus_mono_ns=0,
            pid=1234,
        )
        self.assertEqual(rec, canonical)

    def test_json_nulls_when_unknown(self):
        raw = _raw(16, ts=1, dur=1, tid=1)
        line = self._emit(_events.AttributedEvent(raw, _bpf.API_BY_ID[16], None),
                          json_output=True).strip()
        rec = json.loads(line)
        self.assertIsNone(rec["function"])
        self.assertIsNone(rec["file"])
        self.assertIsNone(rec["line"])
        self.assertIsNone(rec["kernel"])

    def test_event_writes_are_buffered_until_batch_flush(self):
        class TrackingStream(io.StringIO):
            flush_count = 0

            def flush(self):
                self.flush_count += 1
                super().flush()

        stream = TrackingStream()
        joiner = _events.Joiner()
        renderer = _events.Renderer(stream, True, 0, 1234)
        raw = _raw(16, ts=1, dur=1, tid=1)
        renderer.emit(joiner.enrich(
            _events.AttributedEvent(raw, _bpf.API_BY_ID[16], None)
        ))
        self.assertEqual(stream.flush_count, 0)
        renderer.flush()
        self.assertEqual(stream.flush_count, 1)


class DashboardPublisherTest(unittest.TestCase):
    TOKEN = "dashboard-token-" + ("x" * 32)

    @staticmethod
    def _record(**changes):
        record = {
            "timestamp": "2026-08-30T12:10:03.410000+00:00",
            "pid": 1234,
            "tid": 1234,
            "function": "compute",
            "file": "/project/train.py",
            "line": 10,
            "api": "cuLaunchKernel",
            "kernel": "vector_add",
            "return_code": 0,
            "duration_ns": 50_000,
            "details": {"grid": "8,1,1"},
        }
        record.update(changes)
        return record

    @staticmethod
    def _summary(events):
        return {
            "schema_version": 1,
            "complete": True,
            "capture": {
                "events": events,
                "lost_events": 0,
                "dropped_nested_calls": 0,
            },
        }

    def _start_ingest_server(self, token=None):
        state = _web.IngestDashboardState(recent_limit=50, refresh_seconds=0.2)
        server = _web._DashboardServer(
            ("127.0.0.1", 0),
            _web._DashboardRequestHandler,
        )
        server.dashboard_state = state
        server.ingest_token = self.TOKEN if token is None else token
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop_server():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(stop_server)
        return state, server

    def _start_recording_server(self, responder):
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format, *_args):
                return

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                status, response = responder(self.path, body, self.headers)
                encoded = json.dumps(response, separators=(",", ":")).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop_server():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(stop_server)
        return server

    def test_token_is_validated_and_removed(self):
        environment = {"METAGROSS_DASHBOARD_TOKEN": self.TOKEN, "KEEP": "yes"}
        self.assertEqual(_publish.take_dashboard_token(environment), self.TOKEN)
        self.assertEqual(environment, {"KEEP": "yes"})

        invalid = (None, "", "x" * 31, "x" * 129, "has space" * 4, "é" * 32)
        for token in invalid:
            environment = {}
            if token is not None:
                environment["METAGROSS_DASHBOARD_TOKEN"] = token
            with self.subTest(token=token), self.assertRaises(
                _publish.DashboardPublishError
            ):
                _publish.take_dashboard_token(environment)
            self.assertNotIn("METAGROSS_DASHBOARD_TOKEN", environment)

    def test_real_ingest_lifecycle_preserves_local_summary(self):
        state, server = self._start_ingest_server()
        publisher = _publish.DashboardPublisher(
            server.server_port,
            self.TOKEN,
            "workload.py",
        )
        self.addCleanup(publisher.close)
        publisher.start()
        publisher.offer(self._record())
        publisher.offer(
            self._record(
                timestamp="2026-08-30T12:10:04+00:00",
                api="cuMemcpyHtoD",
                kernel=None,
                details={"bytes": 4096},
            )
        )
        local_summary = self._summary(2)

        result = publisher.finish(local_summary)
        payload = state.payload()

        self.assertEqual(result, _publish.PublishResult(0, None))
        self.assertEqual(payload["status"], "COMPLETE")
        self.assertEqual(payload["metrics"]["events"], 2)
        self.assertEqual(payload["metrics"]["delivery_dropped"], 0)
        self.assertEqual(payload["trace_name"], "workload.py")
        self.assertNotIn("delivery_dropped", local_summary["capture"])
        self.assertTrue(local_summary["complete"])

    def test_retry_reuses_identical_batch_and_finishes_in_order(self):
        requests = []
        event_attempts = 0

        def respond(path, body, headers):
            nonlocal event_attempts
            request = json.loads(body)
            requests.append((path, body, headers["Authorization"]))
            base = {
                "schema_version": 1,
                "capture_id": request["capture_id"],
            }
            if path == "/api/capture/start":
                return 200, {**base, "status": "live"}
            if path == "/api/capture/events":
                event_attempts += 1
                if event_attempts == 1:
                    return 503, {"error": "retry"}
                return 200, {**base, "next_sequence": request["sequence"] + 1}
            if path == "/api/capture/finish":
                return 200, {
                    **base,
                    "next_sequence": request["sequence"],
                    "status": "finished",
                }
            return 200, {**base, "status": "aborted"}

        server = self._start_recording_server(respond)
        publisher = _publish.DashboardPublisher(
            server.server_port,
            self.TOKEN,
            "workload.py",
        )
        self.addCleanup(publisher.close)
        with mock.patch.dict(
            os.environ,
            {"HTTP_PROXY": "http://127.0.0.1:1"},
            clear=False,
        ):
            publisher.start()
            publisher.offer(self._record())
            result = publisher.finish(self._summary(1))

        event_bodies = [body for path, body, _auth in requests if path.endswith("events")]
        self.assertEqual(result.dropped_events, 0)
        self.assertEqual(len(event_bodies), 2)
        self.assertEqual(event_bodies[0], event_bodies[1])
        self.assertEqual(
            [path for path, _body, _auth in requests],
            [
                "/api/capture/start",
                "/api/capture/events",
                "/api/capture/events",
                "/api/capture/finish",
            ],
        )
        self.assertTrue(
            all(auth == f"Bearer {self.TOKEN}" for _path, _body, auth in requests)
        )

    def test_queue_overflow_is_declared_only_in_remote_summary(self):
        event_started = threading.Event()
        release_event = threading.Event()
        delivered = []
        finished = []
        publisher = _publish.DashboardPublisher(
            1,
            self.TOKEN,
            "workload.py",
            queue_size=1,
        )

        def post(path, body):
            request = json.loads(body)
            base = {
                "schema_version": 1,
                "capture_id": request["capture_id"],
            }
            if path.endswith("start"):
                return {**base, "status": "live"}
            if path.endswith("events"):
                event_started.set()
                release_event.wait(timeout=2)
                delivered.extend(request["events"])
                return {**base, "next_sequence": request["sequence"] + 1}
            if path.endswith("finish"):
                finished.append(request["summary"])
                return {
                    **base,
                    "next_sequence": request["sequence"],
                    "status": "finished",
                }
            return {**base, "status": "aborted"}

        with mock.patch.object(publisher, "_post", side_effect=post):
            publisher.start()
            publisher.offer(self._record(tid=1))
            self.assertTrue(event_started.wait(timeout=2))
            publisher.offer(self._record(tid=2))
            publisher.offer(self._record(tid=3))
            release_event.set()
            local_summary = self._summary(3)
            result = publisher.finish(local_summary)
        publisher.close()

        self.assertEqual(len(delivered), 2)
        self.assertEqual(result.dropped_events, 1)
        self.assertEqual(finished[0]["capture"]["delivery_dropped"], 1)
        self.assertFalse(finished[0]["complete"])
        self.assertNotIn("delivery_dropped", local_summary["capture"])
        self.assertIsNotNone(publisher.pop_error())
        self.assertIsNone(publisher.pop_error())

    def test_failed_sequence_counts_inflight_queued_and_future_events(self):
        event_started = threading.Event()
        release_event = threading.Event()
        aborted = threading.Event()
        paths = []
        publisher = _publish.DashboardPublisher(
            1,
            self.TOKEN,
            "workload.py",
            queue_size=2,
        )

        def post(path, body):
            request = json.loads(body)
            paths.append(path)
            base = {
                "schema_version": 1,
                "capture_id": request["capture_id"],
            }
            if path.endswith("start"):
                return {**base, "status": "live"}
            if path.endswith("events"):
                event_started.set()
                release_event.wait(timeout=2)
                raise _publish.DashboardPublishError("unacknowledged sequence")
            if path.endswith("abort"):
                aborted.set()
                return {**base, "status": "aborted"}
            raise AssertionError("publisher continued after failed sequence")

        with mock.patch.object(publisher, "_post", side_effect=post):
            publisher.start()
            publisher.offer(self._record(tid=1))
            self.assertTrue(event_started.wait(timeout=2))
            publisher.offer(self._record(tid=2))
            publisher.offer(self._record(tid=3))
            release_event.set()
            self.assertTrue(aborted.wait(timeout=2))
            publisher.offer(self._record(tid=4))
            diagnostic = publisher.pop_error()
            result = publisher.finish(self._summary(4))
        publisher.close()

        self.assertEqual(result.dropped_events, 4)
        self.assertIn("unacknowledged sequence", diagnostic)
        self.assertIsNone(publisher.pop_error())
        self.assertEqual(paths.count("/api/capture/events"), 1)
        self.assertNotIn("/api/capture/finish", paths)

    def test_oversized_record_is_dropped_without_an_event_request(self):
        paths = []
        remote_summary = []
        publisher = _publish.DashboardPublisher(1, self.TOKEN, "workload.py")

        def post(path, body):
            request = json.loads(body)
            paths.append(path)
            base = {
                "schema_version": 1,
                "capture_id": request["capture_id"],
            }
            if path.endswith("start"):
                return {**base, "status": "live"}
            if path.endswith("finish"):
                remote_summary.append(request["summary"])
                return {
                    **base,
                    "next_sequence": request["sequence"],
                    "status": "finished",
                }
            return {**base, "status": "aborted"}

        with mock.patch.object(publisher, "_post", side_effect=post):
            publisher.start()
            publisher.offer(self._record(details={"value": "x" * (64 << 10)}))
            result = publisher.finish(self._summary(1))
        publisher.close()

        self.assertEqual(result.dropped_events, 1)
        self.assertNotIn("/api/capture/events", paths)
        self.assertEqual(remote_summary[0]["capture"]["delivery_dropped"], 1)
        self.assertFalse(remote_summary[0]["complete"])

    def test_shutdown_timeout_is_bounded_and_aborts(self):
        event_started = threading.Event()
        release_event = threading.Event()
        paths = []
        publisher = _publish.DashboardPublisher(
            1,
            self.TOKEN,
            "workload.py",
            shutdown_timeout_s=0.05,
        )

        def post(path, body):
            request = json.loads(body)
            paths.append(path)
            base = {
                "schema_version": 1,
                "capture_id": request["capture_id"],
            }
            if path.endswith("start"):
                return {**base, "status": "live"}
            if path.endswith("events"):
                event_started.set()
                release_event.wait(timeout=2)
                return {**base, "next_sequence": request["sequence"] + 1}
            if path.endswith("abort"):
                return {**base, "status": "aborted"}
            raise AssertionError("timed-out publisher tried to finish")

        with mock.patch.object(publisher, "_post", side_effect=post):
            publisher.start()
            publisher.offer(self._record())
            self.assertTrue(event_started.wait(timeout=2))
            started = time.monotonic()
            result = publisher.finish(self._summary(1))
            elapsed = time.monotonic() - started
            release_event.set()
            publisher.close()

        self.assertLess(elapsed, 0.5)
        self.assertEqual(result.dropped_events, 1)
        self.assertIn("/api/capture/abort", paths)

    def test_wrong_token_fails_start_without_mutating_receiver(self):
        state, server = self._start_ingest_server()
        publisher = _publish.DashboardPublisher(
            server.server_port,
            "wrong-token-" + ("z" * 32),
            "workload.py",
        )
        self.addCleanup(publisher.close)

        with self.assertRaisesRegex(
            _publish.DashboardPublishError,
            "HTTP 401",
        ):
            publisher.start()

        self.assertEqual(state.payload()["status"], "WAITING")
        self.assertEqual(state.payload()["metrics"]["events"], 0)


    def test_ambiguous_event_retry_is_deduplicated_by_real_receiver(self):
        state = _web.IngestDashboardState(recent_limit=50, refresh_seconds=0.05)

        class DropFirstEventAck(_web._DashboardRequestHandler):
            dropped = False

            def _send_ack(self, value):
                if self.path == "/api/capture/events" and not type(self).dropped:
                    type(self).dropped = True
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return
                super()._send_ack(value)
        server = _web._DashboardServer(("127.0.0.1", 0), DropFirstEventAck)
        server.dashboard_state = state
        server.ingest_token = self.TOKEN
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop_server():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(stop_server)
        publisher = _publish.DashboardPublisher(
            server.server_port,
            self.TOKEN,
            "workload.py",
        )
        self.addCleanup(publisher.close)
        publisher.start()
        publisher.offer(self._record())
        result = publisher.finish(self._summary(1))

        self.assertEqual(result.dropped_events, 0)
        self.assertTrue(DropFirstEventAck.dropped)
        self.assertEqual(state.payload()["metrics"]["events"], 1)
        self.assertEqual(state.payload()["status"], "COMPLETE")

    def test_batches_are_ordered_and_limited_to_128_events(self):
        event_started = threading.Event()
        release_event = threading.Event()
        event_requests = []
        publisher = _publish.DashboardPublisher(
            1,
            self.TOKEN,
            "workload.py",
            queue_size=300,
        )

        def post(path, body):
            request = json.loads(body)
            base = {
                "schema_version": 1,
                "capture_id": request["capture_id"],
            }
            if path.endswith("start"):
                return {**base, "status": "live"}
            if path.endswith("events"):
                event_requests.append((body, request))
                if len(event_requests) == 1:
                    event_started.set()
                    release_event.wait(timeout=2)
                return {**base, "next_sequence": request["sequence"] + 1}
            if path.endswith("finish"):
                return {
                    **base,
                    "next_sequence": request["sequence"],
                    "status": "finished",
                }
            return {**base, "status": "aborted"}

        with mock.patch.object(publisher, "_post", side_effect=post):
            publisher.start()
            publisher.offer(self._record(tid=0))
            self.assertTrue(event_started.wait(timeout=2))
            for tid in range(1, 260):
                publisher.offer(self._record(tid=tid))
            release_event.set()
            result = publisher.finish(self._summary(260))
        publisher.close()

        requests = [request for _body, request in event_requests]
        tids = [
            event["tid"]
            for request in requests
            for event in request["events"]
        ]
        self.assertEqual(result.dropped_events, 0)
        self.assertEqual(tids, list(range(260)))
        self.assertTrue(all(len(request["events"]) <= 128 for request in requests))
        self.assertEqual(
            [request["sequence"] for request in requests],
            list(range(len(requests))),
        )
        self.assertTrue(
            all(len(body) <= _publish._MAX_BATCH_BYTES for body, _request in event_requests)
        )

    def test_complete_batch_envelopes_stay_below_one_mibibyte(self):
        event_started = threading.Event()
        release_event = threading.Event()
        event_bodies = []
        publisher = _publish.DashboardPublisher(
            1,
            self.TOKEN,
            "workload.py",
            queue_size=32,
        )

        def post(path, body):
            request = json.loads(body)
            base = {
                "schema_version": 1,
                "capture_id": request["capture_id"],
            }
            if path.endswith("start"):
                return {**base, "status": "live"}
            if path.endswith("events"):
                event_bodies.append(body)
                if len(event_bodies) == 1:
                    event_started.set()
                    release_event.wait(timeout=2)
                return {**base, "next_sequence": request["sequence"] + 1}
            if path.endswith("finish"):
                return {
                    **base,
                    "next_sequence": request["sequence"],
                    "status": "finished",
                }
            return {**base, "status": "aborted"}

        with mock.patch.object(publisher, "_post", side_effect=post):
            publisher.start()
            publisher.offer(self._record(tid=0, details={"value": "x" * 60_000}))
            self.assertTrue(event_started.wait(timeout=2))
            for tid in range(1, 20):
                publisher.offer(
                    self._record(tid=tid, details={"value": "x" * 60_000})
                )
            release_event.set()
            result = publisher.finish(self._summary(20))
        publisher.close()

        self.assertEqual(result.dropped_events, 0)
        self.assertGreaterEqual(len(event_bodies), 3)
        self.assertTrue(
            all(len(body) <= _publish._MAX_BATCH_BYTES for body in event_bodies)
        )

    def test_start_rejects_redirect_large_or_nonexact_acknowledgements(self):
        cases = (
            (
                "redirect",
                302,
                lambda base: base,
                "HTTP 302",
            ),
            (
                "oversized",
                200,
                lambda base: {**base, "status": "live", "padding": "x" * 5000},
                "too large",
            ),
            (
                "extra field",
                200,
                lambda base: {**base, "status": "live", "extra": True},
                "did not match",
            ),
        )
        for name, start_status, response_builder, message in cases:
            paths = []

            def respond(path, body, _headers):
                request = json.loads(body)
                paths.append(path)
                base = {
                    "schema_version": 1,
                    "capture_id": request["capture_id"],
                }
                if path.endswith("abort"):
                    return 200, {**base, "status": "aborted"}
                return start_status, response_builder(base)

            with self.subTest(name=name):
                server = self._start_recording_server(respond)
                publisher = _publish.DashboardPublisher(
                    server.server_port,
                    self.TOKEN,
                    "workload.py",
                )
                with self.assertRaisesRegex(
                    _publish.DashboardPublishError,
                    message,
                ):
                    publisher.start()
                publisher.close()
                self.assertEqual(
                    paths,
                    ["/api/capture/start", "/api/capture/abort"],
                )


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

    def test_root_slash_accepts_any_absolute_path(self):
        # project_root="/" normalizes to root_prefix="/" (not "//"), so
        # any absolute path under it must be accepted rather than every
        # path being rejected.
        p = os.path.join(self.dir.name, "s.py")
        open(p, "w").close()
        self.assertEqual(_validate_script(p, "/"), os.path.realpath(p))


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
    def _start_dashboard_receiver(self, token):
        state = _web.IngestDashboardState(recent_limit=500, refresh_seconds=0.05)
        server = _web._DashboardServer(
            ("127.0.0.1", 0),
            _web._DashboardRequestHandler,
        )
        server.dashboard_state = state
        server.ingest_token = token
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop_server():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(stop_server)
        return state, server

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
        out_name = _fresh_output_path(self)
        proc = subprocess.run(
            ["/usr/bin/python3", "-m", "metagross", "--json", "--output",
             out_name, "--project-root", "/tmp", wrapper],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(out_name) as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        apis = {r["api"] for r in records}
        self.assertTrue(any(a.startswith("cuMemcpyHtoD") for a in apis),
                        f"cudart caller invisible to uprobes; saw {apis}")
        self.assertTrue(any(a.startswith("cuLaunchKernel") for a in apis),
                        f"cudart kernel launch invisible; saw {apis}")

    def _trace_demo(self, *extra):
        out_name = _fresh_output_path(self)
        proc = subprocess.run(
            ["/usr/bin/python3", "-m", "metagross", "--json",
             "--output", out_name, *extra, "examples/gpu_demo.py"],
            capture_output=True, text=True, timeout=120)
        with open(out_name) as handle:
            records = [json.loads(line)
                       for line in handle if line.strip()]
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

    def test_launch_only_without_attribution(self):
        summary_name = _fresh_output_path(self)
        proc, records = self._trace_demo(
            "--trace", "launch", "--no-attribution", "--stats",
            "--summary-output", summary_name,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("vec_add ok", proc.stdout)
        self.assertTrue(records, "no launch records captured")
        self.assertEqual({record["api"] for record in records},
                         {"cuLaunchKernel"})
        self.assertTrue(all(record["function"] is None for record in records))
        self.assertTrue(all(record["file"] is None for record in records))
        self.assertTrue(all(record["line"] is None for record in records))
        self.assertEqual(records[0]["kernel"], "vec_add")
        self.assertIn("metagross: stats", proc.stderr)
        with open(summary_name) as handle:
            summary = json.load(handle)
        self.assertEqual(summary["schema_version"], 1)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["capture"]["events"], len(records))
        self.assertEqual(summary["capture"]["attributed"], 0)
        self.assertEqual(summary["configuration"]["trace_families"], ["launch"])
        self.assertFalse(summary["configuration"]["python_attribution"])
        self.assertEqual(summary["target"]["exit_status"], 0)
        self.assertEqual(summary["top_kernels"][0]["kernel"], "vec_add")

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
        proc.communicate(timeout=60)
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

    def test_direct_dashboard_delivery_is_fileless_and_preserves_target_io(self):
        token = "live-dashboard-token-" + ("x" * 32)
        state, server = self._start_dashboard_receiver(token)
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        os.chown(
            directory,
            int(os.environ.get("SUDO_UID", os.getuid())),
            int(os.environ.get("SUDO_GID", os.getgid())),
        )
        script = os.path.join(directory, "direct_demo.py")
        with open("examples/gpu_demo.py", encoding="utf-8") as source:
            workload = source.read()
        with open(script, "w", encoding="utf-8") as target:
            target.write(
                workload
                + "\nimport os, sys\n"
                + "print('target stderr sentinel', file=sys.stderr)\n"
                + "print('dashboard token present', "
                + "'METAGROSS_DASHBOARD_TOKEN' in os.environ, file=sys.stderr)\n"
            )
        environment = os.environ.copy()
        environment["METAGROSS_DASHBOARD_TOKEN"] = token

        proc = subprocess.run(
            [
                "/usr/bin/python3",
                "-B",
                "-m",
                "metagross",
                "--dashboard-port",
                str(server.server_port),
                "--project-root",
                directory,
                script,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("vec_add ok", proc.stdout)
        self.assertIn("target stderr sentinel", proc.stderr)
        self.assertIn("dashboard token present False", proc.stderr)
        self.assertNotIn(token, proc.stderr)
        payload = state.payload()
        self.assertEqual(payload["status"], "COMPLETE")
        self.assertEqual(payload["metrics"]["delivery_dropped"], 0)
        with state._lock:
            apis = set(state.model.apis)
            functions = {key[0] for key in state.model.functions}
            kernels = set(state.model.kernels)
            summary = state.model.summary
            event_count = state.model.events
        self.assertTrue(any(api.startswith("cuLaunchKernel") for api in apis))
        self.assertTrue(any(api.startswith("cuMemcpyHtoD") for api in apis))
        self.assertTrue(any(api.startswith("cuMemcpyDtoH") for api in apis))
        self.assertTrue(any(api.startswith("cuMemAlloc") for api in apis))
        self.assertTrue(any(api.startswith("cuStreamSynchronize") for api in apis))
        self.assertTrue({"upload", "compute", "download_and_check"} <= functions)
        self.assertIn("vec_add", kernels)
        self.assertEqual(summary["capture"]["events"], event_count)
        self.assertEqual(summary["capture"]["delivery_dropped"], 0)
        self.assertEqual(summary["target"]["exit_status"], 0)
        self.assertEqual(os.listdir(directory), ["direct_demo.py"])

    def test_wrong_dashboard_token_never_releases_target_barrier(self):
        receiver_token = "receiver-live-token-" + ("r" * 32)
        state, server = self._start_dashboard_receiver(receiver_token)
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        os.chown(
            directory,
            int(os.environ.get("SUDO_UID", os.getuid())),
            int(os.environ.get("SUDO_GID", os.getgid())),
        )
        sentinel = os.path.join(directory, "target-ran")
        script = os.path.join(directory, "must_not_run.py")
        with open(script, "w", encoding="utf-8") as target:
            target.write(
                "from pathlib import Path\n"
                f"Path({sentinel!r}).write_text('ran')\n"
                "print('target executed')\n"
            )
        wrong_token = "wrong-live-token-" + ("w" * 32)
        environment = os.environ.copy()
        environment["METAGROSS_DASHBOARD_TOKEN"] = wrong_token

        proc = subprocess.run(
            [
                "/usr/bin/python3",
                "-B",
                "-m",
                "metagross",
                "--dashboard-port",
                str(server.server_port),
                "--project-root",
                directory,
                script,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
        )

        self.assertEqual(proc.returncode, 1)
        self.assertFalse(os.path.exists(sentinel))
        self.assertNotIn("target executed", proc.stdout)
        self.assertIn("cannot start dashboard delivery", proc.stderr)
        self.assertNotIn(receiver_token, proc.stderr)
        self.assertNotIn(wrong_token, proc.stderr)
        self.assertEqual(state.payload()["status"], "WAITING")


def _torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@unittest.skipUnless(INTEGRATION and os.geteuid() == 0 and _torch_available(),
                     "needs integration env and CUDA-enabled torch")
class TorchSmokeTest(unittest.TestCase):
    def test_forward_pass_attributed(self):
        path = "/tmp/metagross_torch.py"
        with open(path, "w") as f:
            f.write(
                "import torch\n"
                "def forward_step(a, b):\n"
                "    return (a @ b).sum().item()\n"
                "a = torch.randn(256, 256, device='cuda')\n"
                "b = torch.randn(256, 256, device='cuda')\n"
                "print('sum', forward_step(a, b))\n")
        self.addCleanup(os.unlink, path)
        out_name = _fresh_output_path(self)
        proc = subprocess.run(
            ["/usr/bin/python3", "-m", "metagross", "--json", "--output",
             out_name, "--project-root", "/tmp", path],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(out_name) as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        launches = [r for r in records if r["api"].startswith("cuLaunchKernel")]
        self.assertTrue(launches, "no kernel launches captured from torch")
        self.assertIn("forward_step", {r["function"] for r in launches})


if __name__ == "__main__":
    unittest.main()
