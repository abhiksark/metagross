# tests/test_metagross.py
"""Tests for metagross. Unprivileged unless RUN_EBPF_INTEGRATION=1."""
import collections
import contextlib
import contextvars
import ctypes
import fcntl
import io
import json
import http.server
import mmap
import os
import runpy
import shutil
import signal
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

_real_in_host_pid_namespace = metagross._in_host_pid_namespace


def setUpModule():
    # The unit suite must also pass inside a container, which has its own
    # PID namespace, and on any architecture. PidNamespaceTest exercises the
    # real namespace check.
    for name, value in (("_in_host_pid_namespace", True), ("_machine", "x86_64")):
        patcher = mock.patch(f"metagross.{name}", return_value=value)
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)


def _reset_seq():
    _profile._seq = 0


def _encode_frame(kind, tid, ts_ns, func, path, line):
    """Encode a FRAME_DEF (id 0) followed by the CALL or RETURN that uses it."""
    return (_profile._encode_frame_def(0, func, path, line)
            + _profile._encode_frame_ref(kind, tid, ts_ns, 0))


def _writer(os_write, **options):
    """A profile writer on a fake descriptor that never sleeps."""
    return _profile._DropCountWriter(
        7, os_write=os_write, wait_writable=lambda fd, timeout_s: None,
        **options)


def _is_project_file(path, project_root):
    return _profile._ProjectClassifier(project_root).includes(path)


def _fresh_output_path(test):
    """Return a not-yet-created trace-output path metagross can create and own.

    The live suite runs as root; pre-creating the file would make it
    root-owned and metagross (dropped to the invoking uid) would rightly
    refuse to truncate it. Letting metagross create the file matches real
    usage and lets it chown the file to the invoker. The directory must
    belong to the invoker too, or metagross refuses to create a file there.
    """
    directory = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, directory, ignore_errors=True)
    os.chown(directory, int(os.environ.get("SUDO_UID", os.getuid())),
             int(os.environ.get("SUDO_GID", os.getgid())))
    return os.path.join(directory, "trace.jsonl")


def _raw(api_id, *, args=(), out=0, name=b"", ret=0, ts=0, dur=0, tid=1):
    ev = _bpf.RawEvent(ts=ts, dur=dur, tid=tid, api_id=api_id, ret=ret,
                       out=out, name=name)
    for i, v in enumerate(args):
        ev.args[i] = v
    return ev


def _stub_attachment():
    return _bpf.Attachment(api=_bpf.API_BY_ID[1], symbol="cuLaunchKernel")


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

    def test_graph_launch_details_name_no_kernel(self):
        # A graph replays many kernels in one call; none of them is "the"
        # kernel, so the row carries the graph and stream handles only.
        api = _bpf.API_BY_ID[21]
        self.assertEqual(api.base, "cuGraphLaunch")
        kernel, det = _events.describe(
            api, _raw(21, args=(0xABC0, 0x77)), self.reg, self.allocs)
        self.assertIsNone(kernel)
        self.assertEqual(det, {"graph_exec": "0xabc0", "stream": "0x77"})

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

    def test_snapshot_aggregates_top_spans_only_when_present(self):
        joiner = _events.Joiner()
        stats = _events.CaptureStats()
        frame = _events.FrameInfo("step", "/p/train.py", 10)
        attributed = [
            _events.AttributedEvent(
                _raw(3, args=(0, 1024), out=0xA, dur=10),
                _bpf.API_BY_ID[3], frame, span="forward",
            ),
            _events.AttributedEvent(
                _raw(7, args=(0xA, 0xB, 512), dur=20),
                _bpf.API_BY_ID[7], frame, span="forward",
            ),
            _events.AttributedEvent(
                _raw(9, args=(0xB, 0xA, 256), ret=1, dur=30),
                _bpf.API_BY_ID[9], frame,  # no enclosing span
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
        self.assertEqual(len(snapshot["top_spans"]), 1)
        self.assertEqual(snapshot["top_spans"][0]["span"], "forward")
        self.assertEqual(snapshot["top_spans"][0]["count"], 2)

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

    def test_lost_profile_records_surfaces_in_capture_and_marks_incomplete(self):
        stats = _events.CaptureStats()
        snapshot = stats.snapshot(
            lost_events=0,
            dropped_nested_calls=0,
            observed_outstanding_bytes=0,
            render_failed=False,
            lost_profile_records=3,
        )
        self.assertFalse(snapshot["complete"])
        self.assertEqual(snapshot["capture"]["lost_profile_records"], 3)

    def test_capture_from_an_unprobed_libcuda_is_incomplete(self):
        snapshot = _events.CaptureStats().snapshot(
            lost_events=0,
            dropped_nested_calls=0,
            observed_outstanding_bytes=0,
            render_failed=False,
            libcuda_mismatch=True,
        )
        self.assertFalse(snapshot["complete"])
        self.assertTrue(snapshot["capture"]["libcuda_mismatch"])

    def test_refused_attributions_make_the_capture_incomplete(self):
        snapshot = _events.CaptureStats().snapshot(
            lost_events=0,
            dropped_nested_calls=0,
            observed_outstanding_bytes=0,
            render_failed=False,
            refused_attributions=2,
        )
        self.assertFalse(snapshot["complete"])
        self.assertEqual(snapshot["capture"]["refused_attributions"], 2)

    def test_lost_profile_records_defaults_to_zero_and_stays_complete(self):
        stats = _events.CaptureStats()
        snapshot = stats.snapshot(
            lost_events=0,
            dropped_nested_calls=0,
            observed_outstanding_bytes=0,
            render_failed=False,
        )
        self.assertTrue(snapshot["complete"])
        self.assertEqual(snapshot["capture"]["lost_profile_records"], 0)


class ParseArgsTest(unittest.TestCase):
    def test_allow_root_target_is_off_by_default(self):
        self.assertFalse(parse_args(["t.py"]).allow_root_target)
        self.assertTrue(
            parse_args(["--allow-root-target", "t.py"]).allow_root_target)

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

    def test_value_options_accept_the_equals_form(self) -> None:
        cfg = parse_args(["--output=/tmp/x.jsonl", "--trace=launch,sync",
                          "--web", "--web-port=0", "script.py", "--output=a"])
        self.assertEqual(cfg.output_path, "/tmp/x.jsonl")
        self.assertEqual(cfg.trace_families, frozenset(("launch", "sync")))
        self.assertEqual(cfg.web_port, 0)
        self.assertEqual(cfg.script_args, ["--output=a"])
        with self.assertRaisesRegex(UsageError, "--output requires a value"):
            parse_args(["--output=", "script.py"])
        with self.assertRaisesRegex(UsageError, "unknown option: --json=1"):
            parse_args(["--json=1", "script.py"])

    def test_single_dash_arguments_are_unknown_options_not_the_script(self):
        # `-V` used to be taken as the script and reported as a missing
        # root privilege.
        for argument in ("-V", "-v", "-m", "-"):
            with self.subTest(argument=argument):
                with self.assertRaisesRegex(UsageError,
                                            f"unknown option: {argument}"):
                    parse_args([argument, "script.py"])
        self.assertEqual(parse_args(["./-odd.py"]).script, "./-odd.py")

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

    def test_web_options_and_target_argument_boundary(self) -> None:
        cfg = parse_args(["--web", "--web-port", "0", "script.py", "--web"])
        self.assertTrue(cfg.web)
        self.assertEqual(cfg.web_port, 0)
        self.assertEqual(cfg.script_args, ["--web"])
        self.assertIsNone(parse_args(["--web", "script.py"]).web_port)

    def test_web_rejects_invalid_combinations(self) -> None:
        for argv, message in (
            (["--web-port", "8000", "script.py"], "requires --web"),
            (["--web", "--dashboard-port", "8765", "script.py"], "cannot be combined"),
            (["--dashboard-port", "8765", "--web", "script.py"], "cannot be combined"),
            (["--web", "--ebpf"], "cannot be combined"),
            (["--ebpf", "--web"], "cannot be combined"),
            (["--web", "--web-port", "65536", "script.py"], "between 0 and 65535"),
            (["--web", "--web-port", "port", "script.py"], "integer"),
        ):
            with self.subTest(argv=argv), self.assertRaisesRegex(UsageError, message):
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
        for option in ("--trace", "--summary-output", "--dashboard-port", "--web-port"):
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
        self.assertIn("--stats", command)

    def test_profile_loss_is_read_from_the_stats_line(self):
        match = self.benchmark["_PROFILE_LOSS_RE"].search(
            "metagross: lost 12 profile records (writer overrun)\n"
            "metagross: stats events=5 attributed=3 unknown=2 errors=0 lost=0 "
            "dropped=0 lost_profile=12 refused=2 complete=false\n")
        self.assertEqual(match.groups(), ("12", "2"))

    def test_summary_counts_loss_and_uses_median(self):
        summary = self.benchmark["_summarize"]([
            {"elapsed_seconds": 3.0, "lost_events": 2,
             "dropped_nested_calls": 0},
            {"elapsed_seconds": 1.0, "lost_events": 0,
             "dropped_nested_calls": 1},
            {"elapsed_seconds": 2.0, "lost_events": 3,
             "dropped_nested_calls": 0, "lost_profile_records": 7,
             "refused_attributions": 4},
        ])
        self.assertEqual(summary["median_seconds"], 2.0)
        self.assertEqual(summary["lost_events"], 5)
        self.assertEqual(summary["dropped_nested_calls"], 1)
        self.assertEqual(summary["lost_profile_records"], 7)
        self.assertEqual(summary["refused_attributions"], 4)
        self.assertIsNone(summary["target_median_seconds"])


@unittest.skipUnless(shutil.which("sh"), "requires a POSIX sh")
class DockerWrapperTest(unittest.TestCase):
    WRAPPER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, "examples", "docker", "metagross")

    def _docker_argv(self, *args, image=None, docker_args=None):
        """Run the wrapper against a fake docker and return docker's argv."""
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            project = os.path.realpath(os.path.join(tmp, "project"))
            os.mkdir(bin_dir)
            os.mkdir(project)
            fake = os.path.join(bin_dir, "docker")
            with open(fake, "w") as f:
                f.write("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            os.chmod(fake, 0o755)
            env = dict(os.environ, PWD=project,
                       PATH=bin_dir + os.pathsep + os.environ["PATH"])
            env.pop("METAGROSS_IMAGE", None)
            env.pop("METAGROSS_DOCKER_ARGS", None)
            if image is not None:
                env["METAGROSS_IMAGE"] = image
            if docker_args is not None:
                env["METAGROSS_DOCKER_ARGS"] = docker_args
            result = subprocess.run(
                ["sh", self.WRAPPER, *args], cwd=project, env=env,
                capture_output=True, text=True, check=True,
            )
        return project, result.stdout.splitlines()

    def test_mounts_and_enters_current_directory(self):
        project, argv = self._docker_argv("run.py", "a b")
        self.assertEqual(argv[argv.index("-w") + 1], project)
        self.assertIn(f"{project}:{project}", argv)
        self.assertEqual(argv[-3:], ["metagross-pytorch", "run.py", "a b"])
        self.assertNotIn("--network=host", argv)

    def test_web_uses_host_network(self):
        _, argv = self._docker_argv("--web", "run.py")
        self.assertIn("--network=host", argv)
        self.assertEqual(argv[-2:], ["--web", "run.py"])

    def test_web_after_the_script_is_a_script_argument(self):
        _, argv = self._docker_argv("run.py", "--web")
        self.assertNotIn("--network=host", argv)
        self.assertEqual(argv[-2:], ["run.py", "--web"])

    def test_an_option_value_ending_in_py_is_not_the_script(self):
        for output in (("--output", "out.py"), ("--output=out.py",)):
            with self.subTest(output=output):
                _, argv = self._docker_argv(*output, "--web", "run.py")
                self.assertIn("--network=host", argv)

    def test_help_version_and_ebpf_run_without_privileges(self):
        for arguments in (["--help"], ["-h"], ["--version"],
                          ["--trace", "launch", "--ebpf"]):
            with self.subTest(arguments=arguments):
                _, argv = self._docker_argv(*arguments)
                self.assertEqual(
                    argv, ["run", "--rm", "-i", "metagross-pytorch", *arguments])

    def test_help_after_the_script_belongs_to_the_script(self):
        _, argv = self._docker_argv("run.py", "--help")
        self.assertIn("--privileged", argv)

    def test_viewers_run_as_the_caller_without_privileges(self):
        project, argv = self._docker_argv("view", "--snapshot", "trace.jsonl")
        for option in ("--privileged", "--pid=host", "--gpus", "--network=host"):
            self.assertNotIn(option, argv)
        self.assertEqual(argv[argv.index("--user") + 1],
                         f"{os.getuid()}:{os.getgid()}")
        self.assertIn(f"{project}:{project}", argv)
        self.assertNotIn("/sys/kernel/debug:/sys/kernel/debug", argv)
        self.assertEqual(argv[-3:], ["view", "--snapshot", "trace.jsonl"])

    def test_web_viewer_uses_host_network(self):
        _, argv = self._docker_argv("view", "--web", "trace.jsonl")
        self.assertIn("--network=host", argv)
        self.assertNotIn("--privileged", argv)

    def test_extra_docker_options_are_passed_through(self):
        _, argv = self._docker_argv(
            "run.py", docker_args="-e CUDA_VISIBLE_DEVICES=1 -v /data:/data:ro")
        image = argv.index("metagross-pytorch")
        self.assertEqual(argv[image - 4:image],
                         ["-e", "CUDA_VISIBLE_DEVICES=1", "-v", "/data:/data:ro"])
        self.assertEqual(argv[image + 1:], ["run.py"])

    def test_image_override(self):
        _, argv = self._docker_argv(image="metagross-pytorch:cu128")
        self.assertEqual(argv[-1], "metagross-pytorch:cu128")


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


class DropPrivilegesTest(unittest.TestCase):
    def test_drops_by_value_in_group_gid_uid_order(self):
        creds = metagross.Credentials(1234, 5678, "fixture", "/home/fixture")
        calls = []
        env = {}
        with mock.patch("metagross.os.getgrouplist", return_value=[5678, 27]), \
                mock.patch("metagross.os.setgroups",
                           side_effect=lambda g: calls.append(("setgroups", tuple(g)))), \
                mock.patch("metagross.os.setgid",
                           side_effect=lambda g: calls.append(("setgid", g))), \
                mock.patch("metagross.os.setuid",
                           side_effect=lambda u: calls.append(("setuid", u))), \
                mock.patch.dict("metagross.os.environ", env, clear=False):
            metagross._drop_privileges(creds)
            # Assert HOME while the environ patch is still active: patch.dict
            # reverts the dict on __exit__, so this must run inside the block.
            self.assertEqual(metagross.os.environ["HOME"], "/home/fixture")
        self.assertEqual(calls, [
            ("setgroups", (5678, 27)),   # groups from getgrouplist(user, gid)
            ("setgid", 5678),            # gid by value
            ("setuid", 1234),            # uid by value, last (irreversible after)
        ])


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

    def test_existing_file_with_another_hard_link_is_preserved(self):
        # Truncating it would also empty the file under its other name.
        path, other = self._path("out.jsonl"), self._path("other-name")
        with open(path, "wb") as f:
            f.write(b"keep this")
        os.link(path, other)
        with self.assertRaisesRegex(MetagrossError, "hard link"):
            open_trace_output(path, self.uid, self.gid)
        with open(other, "rb") as f:
            self.assertEqual(f.read(), b"keep this")

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

        def dev_directory(_path, _uid):
            # /dev is refused as a directory that is not the caller's; step
            # past that to reach the check on the file itself.
            return real_open("/dev", os.O_RDONLY | os.O_DIRECTORY)

        with mock.patch("metagross.os.open", side_effect=record_open), \
                mock.patch("metagross._open_output_parent",
                           side_effect=dev_directory):
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

    def test_private_folder_under_a_group_writable_one_is_refused(self):
        # With umask 002 a project folder is group-writable, and a private
        # subfolder inside it does not make the path safe. The message must
        # name the folder to fix.
        project = self._path("project")
        os.mkdir(project)
        os.chmod(project, 0o775)
        traces = os.path.join(project, "traces")
        os.mkdir(traces, 0o700)
        path = os.path.join(traces, "trace.jsonl")
        with self.assertRaises(MetagrossError) as caught:
            with open_trace_output(path, self.uid, self.gid):
                pass
        self.assertIn(repr(os.path.realpath(project)), str(caught.exception))
        self.assertIn("chmod go-w", str(caught.exception))
        self.assertFalse(os.path.exists(path))
        os.chmod(project, 0o755)
        with open_trace_output(path, self.uid, self.gid):
            pass
        self.assertTrue(os.path.exists(path))

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
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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
        self._run_with_outputs(trace, summary, "has another hard link")
        with open(trace, "rb") as handle:
            self.assertEqual(handle.read(), b"keep this capture")

    def test_same_path_for_both_outputs_is_rejected(self):
        trace = self._path("trace")
        with open(trace, "wb") as handle:
            handle.write(b"keep this capture")
        self._run_with_outputs(trace, trace, "must be different files")
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
            self._run_with_outputs(trace, summary, "has another hard link")
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

    def test_output_directory_must_be_the_callers_or_a_shared_sticky_one(self):
        # Root creates the file and gives it to the caller. In a root-owned
        # directory that would hand an unprivileged caller a file in, say,
        # /etc/ld.so.conf.d.
        if self.uid != 0:
            for directory in ("/etc", "/usr/lib"):
                with self.subTest(directory=directory):
                    with self.assertRaisesRegex(
                            MetagrossError, "does not belong to you"):
                        metagross._open_output_parent(
                            f"{directory}/metagross-test.conf", self.uid)
        os.close(metagross._open_output_parent("/tmp/trace.jsonl", self.uid))
        os.close(metagross._open_output_parent(self._path("trace"), self.uid))

    @unittest.skipUnless(os.geteuid() == 0, "needs root for real output ownership")
    def test_root_does_not_create_a_callers_file_in_a_root_directory(self):
        path = self._path("trace")  # the directory belongs to root
        with self.assertRaisesRegex(MetagrossError, "does not belong to you"):
            with open_trace_output(path, 1000, 1000):
                pass
        self.assertFalse(os.path.exists(path))

    @unittest.skipUnless(os.geteuid() == 0, "needs root for real output ownership")
    def test_real_foreign_inode_swap_preserves_contents(self):
        os.chown(self.dir.name, 1000, 1000)
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
        os.chown(self.dir.name, 1000, 1000)
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
        self.assertEqual(
            categories, {"launch", "launch_ex", "graph_launch", "register"})
        source = _bpf.build_source(4242, selected)
        self.assertIn("enter_cuGraphLaunch", source)
        self.assertIn("exit_cuGraphLaunch", source)

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

    def test_mapped_libcuda_lists_the_driver_files_in_a_maps_listing(self):
        maps = (
            "55d0c0a00000-55d0c0a01000 r--p 00000000 08:02 131 /usr/bin/python3.10\n"
            "7f10a0000000-7f10a0021000 rw-p 00000000 00:00 0 \n"
            "7f10b0000000-7f10b0400000 r-xp 00000000 08:02 977 "
            "/opt/nvidia driver/libcuda.so.550.54.14\n"
            "7f10b0400000-7f10b0500000 rw-p 00400000 08:02 977 "
            "/opt/nvidia driver/libcuda.so.550.54.14\n"
            "7f10c0000000-7f10c0100000 r-xp 00000000 08:02 978 "
            "/usr/lib/x86_64-linux-gnu/libcudart.so.12\n"
            "7ffd5a1f0000-7ffd5a211000 rw-p 00000000 00:00 0 [stack]\n"
        )
        self.assertEqual(_bpf.mapped_libcuda(maps),
                         {"/opt/nvidia driver/libcuda.so.550.54.14": 977})
        self.assertEqual(_bpf.mapped_libcuda(""), {})

    def _mapped(self, path):
        """Map a file into this process, as a loaded library would be."""
        stack = contextlib.ExitStack()
        handle = stack.enter_context(open(path, "rb"))
        stack.enter_context(mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ))
        return stack

    def _library_file(self, *parts):
        path = os.path.join(*parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"\0" * 4096)
        return path

    def test_target_that_loaded_another_libcuda_gets_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = self._library_file(tmp, "libcuda.so.550.54.14")
            other = self._library_file(tmp, "elsewhere", "libcuda.so.1")
            self.assertEqual(_bpf.loaded_libcuda(os.getpid()), {})
            with self._mapped(loaded):
                mapped = _bpf.loaded_libcuda(os.getpid())
            self.assertEqual(
                mapped, {os.path.realpath(loaded): os.stat(loaded).st_ino})
            self.assertIsNone(metagross._libcuda_mismatch_warning({}, other))
            self.assertIsNone(metagross._libcuda_mismatch_warning(mapped, loaded))
            self.assertEqual(
                metagross._libcuda_mismatch_warning(mapped, other),
                f"metagross: the script loaded {os.path.realpath(loaded)}, but "
                f"the probes are on {os.path.realpath(other)}; its CUDA calls "
                "are not traced")

    def test_libcuda_check_survives_a_mapped_file_with_a_non_utf8_name(self):
        # The check is a diagnostic. An error here would end the capture and
        # kill the script.
        with tempfile.TemporaryDirectory() as tmp:
            odd = os.path.join(os.fsencode(tmp), b"data-\xff\xfe.bin")
            with open(odd, "wb") as handle:
                handle.write(b"\0" * 4096)
            loaded = self._library_file(tmp, "libcuda.so.1")
            with self._mapped(odd), self._mapped(loaded):
                self.assertEqual(set(_bpf.loaded_libcuda(os.getpid())),
                                 {os.path.realpath(loaded)})

    def test_probed_libcuda_reached_by_another_path_is_not_a_mismatch(self):
        # uprobes follow the file, not the name it was opened by.
        with tempfile.TemporaryDirectory() as tmp:
            probed = self._library_file(tmp, "a", "libcuda.so.1")
            other_name = os.path.join(tmp, "b", "libcuda.so.1")
            os.makedirs(os.path.dirname(other_name))
            os.link(probed, other_name)
            with self._mapped(other_name):
                mapped = _bpf.loaded_libcuda(os.getpid())
            self.assertEqual(set(mapped), {os.path.realpath(other_name)})
            self.assertIsNone(
                metagross._libcuda_mismatch_warning(mapped, probed))

    def test_libcuda_warning_keeps_escape_sequences_off_the_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = self._library_file(
                os.fsencode(tmp), b"x\x1b[2Jy\xff", b"libcuda.so.1")
            with self._mapped(loaded):
                mapped = _bpf.loaded_libcuda(os.getpid())
            self.assertEqual(len(mapped), 1)
            warning = metagross._libcuda_mismatch_warning(
                mapped, "/nonexistent/libcuda.so.1")
            self.assertIn("/x?[2Jy?/libcuda.so.1", warning)
            self.assertNotIn("\x1b", warning)
            warning.encode("utf-8")  # printable: no lone surrogates

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
        # Block Z, shared memory and the stream are the 7th to 9th arguments
        # and live on the caller's stack.
        start = self.src.index("int enter_cuLaunchKernel(")
        body = self.src[start:self.src.index("int exit_cuLaunchKernel(")]
        for slot, offset in ((6, 8), (7, 16), (8, 24)):
            self.assertIn(
                f"bpf_probe_read_user(&f.args[{slot}], sizeof(u64), "
                f"(void *)(sp + {offset}));", body)

    def test_launch_ex_reads_the_config_struct(self):
        start = self.src.index("int enter_cuLaunchKernelEx(")
        body = self.src[start:self.src.index("int exit_cuLaunchKernelEx(")]
        self.assertIn("bpf_probe_read_user(&dims, sizeof(dims), cfg);", body)
        self.assertIn("bpf_probe_read_user(&shmem, sizeof(shmem), cfg + 24);", body)
        self.assertIn("bpf_probe_read_user(&stream, sizeof(stream), cfg + 32);", body)

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

    def test_decode_event_without_a_name_field(self):
        # Only the name-reading probes send the long record; every other
        # event stops before the name.
        raw = _bpf.RawEvent(ts=7, dur=9, tid=5, api_id=3, ret=0, out=0x9000)
        short = bytes(bytearray(raw))[:_bpf.RawEvent.name.offset]
        self.assertEqual(len(short), 112)
        ev = _bpf.decode_event(short)
        self.assertLess(ctypes.sizeof(ev), 128)  # no room kept for a name
        self.assertEqual((ev.ts, ev.api_id, ev.out), (7, 3, 0x9000))
        self.assertEqual(ev.name, b"")

    def test_decode_event_keeps_a_long_kernel_name(self):
        name = b"_ZN2at6native" + b"x" * 600
        raw = _bpf.RawEvent(ts=1, api_id=18, out=0xF00, name=name)
        ev = _bpf.decode_event(bytes(bytearray(raw)))
        self.assertEqual(ev.name, name)

    def test_only_name_reading_probes_reserve_the_long_record(self):
        src = _bpf.build_source(4242)
        self.assertIn("#define NAME_MAX_LEN 1024", src)
        blocks = src.split("\nint exit_")[1:]
        long_record = {block.split("(", 1)[0] for block in blocks
                       if "sizeof(struct name_event_t)" in block}
        self.assertEqual(long_record,
                         {"cuModuleGetFunction", "cuLibraryGetKernel"})
        for block in blocks:
            base = block.split("(", 1)[0]
            if base not in long_record:
                self.assertIn("sizeof(struct event_t)", block)
                self.assertNotIn("->name", block)

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
    def setUp(self):
        _reset_seq()

    def _reader_after_hello(self):
        """A RecordReader past its HELLO, so plain frame assertions below
        do not have to account for the reader's own gap bookkeeping."""
        reader = _profile.RecordReader()
        self.assertEqual(reader.feed(_profile.encode_hello(pid=1, start_ns=0)), [])
        return reader

    def test_roundtrip_single(self):
        reader = self._reader_after_hello()
        blob = _encode_frame(_profile.CALL, 7, 123456789,
                                     "train_step", "/p/train.py", 31)
        self.assertEqual(reader.feed(blob),
                         [("frame", _profile.CALL, 7, 123456789, "train_step",
                           "/p/train.py", 31)])

    def test_split_feed(self):
        # Split inside the 6-byte common header: feed() must not even try
        # to unpack rtype/seq yet.
        reader = self._reader_after_hello()
        blob = _encode_frame(_profile.RETURN, 7, 99, "f", "/p/a.py", 2)
        self.assertEqual(reader.feed(blob[:5]), [])
        self.assertEqual(reader.feed(blob[5:]),
                         [("frame", _profile.RETURN, 7, 99, "f", "/p/a.py", 2)])

    def test_split_feed_inside_variable_length_body(self):
        # Split after the header and fixed prefix (whose seq is now known)
        # but before the func/path bytes finish arriving. The seq check
        # must not fire on the first, incomplete feed -- only once the
        # whole record is present -- or a slow/chunked write would
        # manufacture a spurious gap on its own record. _encode_frame's blob
        # is FRAME_DEF-first, so the split lands inside FRAME_DEF's own
        # func/path bytes (the trailing CALL/RETURN reference is untouched
        # until the second feed).
        reader = self._reader_after_hello()
        blob = _encode_frame(_profile.RETURN, 7, 99,
                                     "somewhat_longer_func_name",
                                     "/p/a/longer/path.py", 2)
        split = 6 + _profile._FRAME_DEF_PREFIX.size + 4  # a few bytes into func
        self.assertLess(split, len(blob))
        self.assertEqual(reader.feed(blob[:split]), [])
        self.assertEqual(reader.feed(blob[split:]),
                         [("frame", _profile.RETURN, 7, 99,
                           "somewhat_longer_func_name",
                           "/p/a/longer/path.py", 2)])

    def test_multiple_records_one_feed(self):
        reader = self._reader_after_hello()
        blob = (_encode_frame(_profile.CALL, 1, 1, "a", "/p/a.py", 1)
                + _encode_frame(_profile.RETURN, 1, 2, "a", "/p/a.py", 1))
        self.assertEqual(len(reader.feed(blob)), 2)

    def test_truncated_multibyte_does_not_raise(self):
        # "x" + "e"-acute * 300 is 601 bytes; the raw 500-byte truncation in
        # encode_frame cuts the 500th byte inside a two-byte "e"-acute
        # character (byte 499 is its leading 0xC3), so a strict utf-8
        # decode of the truncated bytes raises UnicodeDecodeError. feed()
        # must not raise; it replaces the partial character and still
        # yields exactly one record.
        reader = self._reader_after_hello()
        long_str = "x" + "é" * 300
        blob = _encode_frame(_profile.CALL, 3, 42, long_str,
                                     long_str, 5)
        records = reader.feed(blob)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0][4].endswith("�"))
        self.assertTrue(records[0][5].endswith("�"))

    def test_hello_yields_nothing_and_sets_version(self):
        reader = _profile.RecordReader()
        self.assertEqual(
            reader.feed(_profile.encode_hello(pid=4321, start_ns=99)), [])
        self.assertEqual(reader.version, 4)

    def test_missing_hello_emits_gap_then_frame(self):
        # No HELLO fed first: _expected starts as None, so the very first
        # record (even at its own seq 0) is treated as a gap rather than a
        # silently trusted unannounced stream. With frame interning, the
        # very first wire record for any frame is its FRAME_DEF, which
        # carries no ts_ns field -- so the gap is held pending rather than
        # fired with an unknown ts, and is reported with the ts of the
        # next record that DOES carry a real one: here, the CALL's ts=5.
        # (If the stream had ended before any real-ts record arrived, it
        # would fall back to ("gap", None) via RecordReader.finalize().)
        blob = _encode_frame(_profile.CALL, 1, 5, "f", "/p/a.py", 1)
        out = _profile.RecordReader().feed(blob)
        self.assertEqual(out[0], ("gap", 5))
        self.assertEqual(out[1], ("frame", _profile.CALL, 1, 5, "f", "/p/a.py", 1))

    def test_record_type_reads_rtype_byte(self):
        hello = _profile.encode_hello(pid=1, start_ns=0)
        # _encode_frame's blob is FRAME_DEF-first; record_type reads that
        # leading record's rtype, not the CALL/RETURN reference behind it.
        frame = _encode_frame(_profile.CALL, 1, 1, "f", "/p/a.py", 1)
        self.assertEqual(_profile.record_type(hello), _profile.HELLO)
        self.assertEqual(_profile.record_type(frame), _profile.FRAME_DEF)

    def test_span_roundtrip(self):
        reader = self._reader_after_hello()
        named = _profile.encode_span(9, 100, "fwd")
        none = _profile.encode_span(9, 150, None)
        self.assertEqual(reader.feed(named), [("span", 9, 100, "fwd")])
        self.assertEqual(reader.feed(none), [("span", 9, 150, None)])

    def test_hook_replaced_is_a_gap_at_its_own_timestamp(self):
        reader = self._reader_after_hello()
        frame = _encode_frame(_profile.CALL, 1, 100, "f", "/p/a.py", 1)
        replaced = _profile.encode_hook_replaced(250)
        self.assertEqual(reader.feed(frame + replaced),
                         [("frame", _profile.CALL, 1, 100, "f", "/p/a.py", 1),
                          ("gap", 250)])
        self.assertEqual(reader.hook_replacements, 1)
        self.assertEqual(reader.lost_records, 1)

    def test_end_record_closes_the_stream_cleanly(self):
        reader = self._reader_after_hello()
        frame = _encode_frame(_profile.CALL, 1, 100, "f", "/p/a.py", 1)
        end = _profile.encode_end(300)
        self.assertEqual(reader.feed(frame + end[:5]),
                         [("frame", _profile.CALL, 1, 100, "f", "/p/a.py", 1)])
        self.assertEqual(reader.feed(end[5:]), [])
        self.assertEqual(reader.end_of_stream(), [])
        self.assertFalse(reader.ended_early)
        self.assertEqual(reader.lost_records, 0)

    def test_end_record_after_dropped_records_is_a_gap(self):
        reader = self._reader_after_hello()
        _encode_frame(_profile.CALL, 1, 100, "f", "/p/a.py", 1)  # dropped
        self.assertEqual(reader.feed(_profile.encode_end(300)), [("gap", 300)])
        self.assertEqual(reader.end_of_stream(), [])
        self.assertEqual(reader.lost_records, 2)  # the DEF and the CALL

    def test_oversized_name_length_is_a_corrupt_stream(self):
        # The hook never writes a name longer than _MAX_STR. A longer length
        # can only come from a target writing to the pipe itself; the reader
        # must not wait for, or keep, that much data.
        reader = self._reader_after_hello()
        big = _profile._MAX_STR + 1
        header = _profile._COMMON.pack(_profile.FRAME_DEF, reader._expected, 0)
        definition = _profile._FRAME_DEF_PREFIX.pack(0, 1, 0, big, 1)
        self.assertEqual(reader.feed(header + definition), [("gap", None)])
        self.assertEqual(reader._buf, b"")
        reader = self._reader_after_hello()
        header = _profile._COMMON.pack(_profile.SPAN_SET, reader._expected, 0)
        named = _profile._SPAN_SET_PREFIX.pack(1, 10, big)
        self.assertEqual(reader.feed(header + named), [("gap", None)])

    def test_frame_definitions_are_capped(self):
        reader = self._reader_after_hello()
        with mock.patch.object(_profile, "_MAX_FRAMES", 2):
            for frame_id in range(3):
                reader.feed(_profile._encode_frame_def(
                    frame_id, f"f{frame_id}", "/p/a.py", frame_id))
            out = reader.feed(_profile._encode_frame_ref(_profile.CALL, 1, 10, 2))
        self.assertEqual(len(reader._frames), 2)
        self.assertEqual(out, [("gap", 10)])   # frame 2 was never kept
        self.assertEqual(reader.lost_records, 1)

    def test_call_from_a_frame_past_the_cap_is_not_given_to_its_caller(self):
        reader = self._reader_after_hello()
        joiner = _events.Joiner(hold_ns=100)
        with mock.patch.object(_profile, "_MAX_FRAMES", 1):
            data = (_profile._encode_frame_def(0, "f0", "/p/a.py", 1)
                    + _profile._encode_frame_def(1, "f1", "/p/a.py", 2)
                    + _profile._encode_frame_ref(_profile.CALL, 7, 10, 0)
                    + _profile._encode_frame_ref(_profile.CALL, 7, 20, 1))
            for rec in reader.feed(data):
                joiner.on_profile_record(rec)
        joiner.on_gpu_event(_raw(16, ts=30, dur=5, tid=7))   # made by f1
        (event,) = joiner.flush(now_ns=1000)
        self.assertIsNone(event.frame)

    def test_unknown_rtype_does_not_raise(self):
        reader = self._reader_after_hello()
        garbage = _profile._COMMON.pack(99, reader._expected, 0) + b"\x00" * 8
        out = reader.feed(garbage)
        self.assertEqual(out, [("gap", None)])
        # The bad bytes were dropped; feeding nothing more yields nothing.
        self.assertEqual(reader.feed(b""), [])

    def test_unknown_frame_id_drops_one_record_without_clearing_map(self):
        # A CALL/RETURN can reference an id the reader does not hold only
        # past the frame cap or on a corrupt stream. The reader must fail
        # closed -- drop
        # that one record and count it, never guess a frame, and never
        # clear the map on the strength of one bad id.
        reader = self._reader_after_hello()
        known = _encode_frame(_profile.CALL, 7, 10, "run", "/p/a.py", 1)
        self.assertEqual(len(reader.feed(known)), 1)  # frame_id 0 now defined
        bad = _profile._encode_frame_ref(_profile.CALL, tid=7, ts_ns=20,
                                         frame_id=999)
        self.assertEqual(reader.feed(bad), [("gap", 20)])
        self.assertEqual(reader.lost_records, 1)
        # The map survived the drop: a later reference to the earlier,
        # real frame_id (0) still resolves correctly.
        again = _profile._encode_frame_ref(_profile.RETURN, tid=7, ts_ns=30,
                                           frame_id=0)
        self.assertEqual(reader.feed(again),
                         [("frame", _profile.RETURN, 7, 30, "run", "/p/a.py", 1)])


class FrameInterningTest(unittest.TestCase):
    def setUp(self):
        _reset_seq()

    def test_frame_defined_once_then_referenced_by_id(self):
        written = []
        emitter = _profile._FrameEmitter(written.append)  # write = collect bytes
        emitter.emit(_profile.CALL, 7, 10, "run", "/p/a.py", 1)
        emitter.emit(_profile.CALL, 7, 20, "run", "/p/a.py", 1)  # same frame
        # Exactly one FRAME_DEF was written across the two same-frame calls.
        # types.count uses the wire rtype (_FRAME_CALL_RTYPE), not the
        # CALL/RETURN "kind" constant (0/1): those are a separate value
        # space (kind 0/RETURN kind 1 collide with HELLO/FRAME_DEF rtypes),
        # kept apart deliberately -- see the module docstring near CALL/
        # RETURN and _FRAME_CALL_RTYPE.
        types = [_profile.record_type(b) for b in written]
        self.assertEqual(types.count(_profile.FRAME_DEF), 1)
        self.assertEqual(types.count(_profile._FRAME_CALL_RTYPE), 2)
        # The reader reconstructs identical FrameInfo fields for both references.
        reader = _profile.RecordReader()
        frames = []
        for b in written:
            frames += [r for r in reader.feed(b) if r[0] == "frame"]
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0][4:], ("run", "/p/a.py", 1))
        self.assertEqual(frames[1][4:], ("run", "/p/a.py", 1))


class FrameEmitterConcurrencyTest(unittest.TestCase):
    def setUp(self):
        _reset_seq()

    def test_concurrent_emit_and_span_do_not_desync_seq(self):
        # emit() and span() both run on the threading.setprofile hook, which
        # fires on every target thread. Without a single
        # lock covering _next_seq() + write() for both paths, a race
        # between them interleaves or duplicates seqs on the wire and the
        # reader manufactures false gaps. list.append is itself atomic
        # under the GIL, so the order records land in `written` reflects
        # the order the lock let them through -- feeding them back in that
        # same order must decode with zero gaps if the lock is effective.
        #
        # The default GIL switch interval rarely lands a context switch
        # inside the tiny window between _next_seq()'s read and write of
        # the module-global counter, so this test forces very frequent
        # switches to make the race actually observable -- verified against
        # a deliberately unlocked `_FrameEmitter` (reverting the `with
        # self._lock:` wrapping) to reproduce dozens of manufactured gaps
        # within a few thousand iterations; this test must stay green with
        # the lock in place.
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            written = []
            emitter = _profile._FrameEmitter(written.append)
            hello_bytes = _profile.encode_hello(pid=1, start_ns=0)  # consumes seq 0

            def emit_worker():
                for i in range(3000):
                    emitter.emit(_profile.CALL, 1, i, "f", "/p/a.py", 1)

            def span_worker():
                for i in range(3000):
                    emitter.span(2, i, "op")
                    emitter.span(2, i, None)

            threads = [threading.Thread(target=emit_worker),
                       threading.Thread(target=span_worker)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            sys.setswitchinterval(old_interval)

        reader = _profile.RecordReader()
        records = reader.feed(hello_bytes)
        for data in written:
            records += reader.feed(data)
        self.assertNotIn("gap", [rec[0] for rec in records])
        self.assertEqual(reader.lost_records, 0)

    def test_span_holds_the_lock_across_the_write(self):
        emitter = _profile._FrameEmitter(lambda data: None)
        held = []

        def write(data):
            held.append(emitter._lock.locked())

        emitter._write = write
        emitter.span(1, 10, "op")
        emitter.span(1, 20, None)
        self.assertEqual(held, [True, True])


class ProfileDropTest(unittest.TestCase):
    def setUp(self):
        _reset_seq()

    def test_current_emitter_is_none_when_profiling_is_not_installed(self):
        # install() only ever runs in a forked, exec'd target child; the
        # test process itself never calls it, so current_emitter() must
        # not resolve to some leftover module-level state.
        self.assertIsNone(_profile.current_emitter())

    def test_frame_def_never_drops_but_calls_do(self):
        seen = []

        def fake_write(fd, data):
            if _profile.record_type(data) == _profile._FRAME_CALL_RTYPE:
                raise BlockingIOError()   # pipe full for ordinary records
            seen.append(_profile.record_type(data))  # DEF must reach here
            return len(data)

        writer = _writer(fake_write)
        emitter = _profile._FrameEmitter(writer.write)
        emitter.emit(_profile.CALL, 1, 10, "f", "/p/a.py", 1)
        self.assertEqual(writer.dropped, 1)               # the CALL dropped
        self.assertIn(_profile.FRAME_DEF, seen)            # the DEF survived

    def test_each_dropped_record_is_added_to_the_drop_counter(self):
        def full_pipe(fd, data):
            raise BlockingIOError()

        drops_fd = os.eventfd(0, os.EFD_NONBLOCK)
        self.addCleanup(os.close, drops_fd)
        clock = iter([0.0, 5.0]).__next__  # the bounded wait times out at once
        writer = _writer(full_pipe, clock=clock, drops_fd=drops_fd)
        emitter = _profile._FrameEmitter(writer.write)
        emitter.span(1, 10, "step")                              # ordinary
        emitter.emit(_profile.CALL, 1, 20, "f", "/p/a.py", 1)    # its FRAME_DEF
        self.assertEqual(writer.dropped, 2)
        self.assertEqual(os.eventfd_read(drops_fd), 2)

    def test_hook_replaced_is_never_dropped(self):
        attempts = []

        def flaky_write(fd, data):
            attempts.append(1)
            if len(attempts) < 3:
                raise BlockingIOError()
            return len(data)

        writer = _writer(flaky_write)
        _profile._FrameEmitter(writer.write).hook_replaced(10)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(writer.dropped, 0)

    def test_end_record_is_retried_like_a_frame_definition(self):
        attempts = []

        def flaky_write(fd, data):
            attempts.append(1)
            if len(attempts) < 3:
                raise BlockingIOError()
            return len(data)

        writer = _writer(flaky_write)
        _profile._FrameEmitter(writer.write).end(10)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(writer.dropped, 0)

    def test_frame_def_retries_through_blocking_io_error_until_it_fits(self):
        attempts = []

        def flaky_write(fd, data):
            attempts.append(1)
            if len(attempts) < 3:
                raise BlockingIOError()
            return len(data)

        writer = _writer(flaky_write)
        emitter = _profile._FrameEmitter(writer.write)
        emitter.emit(_profile.CALL, 1, 10, "f", "/p/a.py", 1)
        # The FRAME_DEF retried until it fit (3 attempts); the CALL then
        # succeeds on the fd's first (4th overall) try.
        self.assertEqual(len(attempts), 4)
        self.assertEqual(writer.dropped, 0)

    def test_frame_def_gives_up_when_the_pipe_stays_full(self):
        ticks = iter(range(1000))
        full = [True]
        written = []

        def stalled_write(fd, data):
            if full[0]:
                raise BlockingIOError()
            written.append(data)
            return len(data)

        writer = _writer(stalled_write, clock=lambda: float(next(ticks)))
        emitter = _profile._FrameEmitter(writer.write)
        emitter.emit(_profile.CALL, 1, 10, "f", "/p/a.py", 1)  # must return
        self.assertEqual(writer.dropped, 1)  # the DEF; its CALL was not sent

        full[0] = False
        emitter.emit(_profile.CALL, 1, 20, "f", "/p/a.py", 1)
        self.assertEqual(
            [_profile.record_type(data) for data in written],
            [_profile.FRAME_DEF, _profile._FRAME_CALL_RTYPE])
        # The reader sees the hole and fails closed before the new frame.
        reader = _profile.RecordReader()
        reader._expected = _profile._COMMON.unpack_from(written[0])[1] - 1
        records = reader.feed(b"".join(written))
        self.assertEqual(records, [
            ("gap", 20), ("frame", _profile.CALL, 1, 20, "f", "/p/a.py", 1)])
        self.assertEqual(reader.lost_records, 1)

    def test_stalled_pipe_is_waited_for_only_once(self):
        attempts = []

        def stalled_write(fd, data):
            attempts.append(1)
            raise BlockingIOError()

        ticks = iter(range(1000))
        writer = _writer(stalled_write, clock=lambda: next(ticks) / 4)
        emitter = _profile._FrameEmitter(writer.write)
        emitter.emit(_profile.CALL, 1, 10, "f", "/p/a.py", 1)
        waited = len(attempts)
        self.assertGreater(waited, 1)
        emitter.emit(_profile.CALL, 1, 20, "g", "/p/a.py", 2)
        self.assertEqual(len(attempts), waited + 1)  # one try, no second wait
        self.assertEqual(writer.dropped, 2)

    def test_a_slow_controller_costs_one_wait_per_cooldown(self):
        # The controller reads a little now and then: a definition gets
        # through, the pipe fills again. That must not cost the script a
        # full wait for every function it calls for the first time.
        now = [0.0]
        full = [True]
        waits = []

        def write(fd, data):
            if full[0]:
                raise BlockingIOError()
            return len(data)

        def wait(fd, timeout_s):
            waits.append(timeout_s)
            now[0] += timeout_s

        writer = _profile._DropCountWriter(
            7, os_write=write, clock=lambda: now[0], wait_writable=wait)
        emitter = _profile._FrameEmitter(writer.write)
        emitter.emit(_profile.CALL, 1, 10, "a", "/p/a.py", 1)
        self.assertEqual(waits, [_profile._MUST_DELIVER_TIMEOUT_S])
        full[0] = False
        emitter.emit(_profile.CALL, 1, 20, "b", "/p/a.py", 2)  # gets through
        full[0] = True
        emitter.emit(_profile.CALL, 1, 30, "c", "/p/a.py", 3)
        self.assertEqual(len(waits), 1)                        # no second wait
        now[0] += _profile._STALL_COOLDOWN_S
        emitter.emit(_profile.CALL, 1, 40, "d", "/p/a.py", 4)
        self.assertEqual(len(waits), 2)                        # allowed again

    def test_writer_stops_for_good_when_its_descriptor_fails(self):
        # EBADF: the script closed the number. It may reuse it for a file of
        # its own, so the writer must never touch it again.
        attempts = []

        def closed_write(fd, data):
            attempts.append(data)
            raise OSError(9, "Bad file descriptor")

        writer = _writer(closed_write)
        emitter = _profile._FrameEmitter(writer.write)
        emitter.span(1, 10, "step")
        self.assertTrue(writer.closed)
        emitter.emit(_profile.CALL, 1, 20, "f", "/p/a.py", 1)
        emitter.end(30)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(writer.dropped, 0)

    def test_writer_stops_when_the_number_names_another_file(self):
        # The script closed the descriptor and opened a file that was given
        # the same number: writes would succeed, into the script's file.
        written = []
        ours = [True]
        writer = _writer(lambda fd, data: written.append(data),
                         still_ours=lambda: ours[0])
        emitter = _profile._FrameEmitter(writer.write)
        for _ in range(_profile._IDENTITY_CHECK_EVERY * 2):
            emitter.span(1, 10, "step")
        self.assertFalse(writer.closed)
        ours[0] = False
        before = len(written)
        for _ in range(_profile._IDENTITY_CHECK_EVERY * 3):
            emitter.span(1, 10, "step")
        self.assertTrue(writer.closed)
        self.assertLess(len(written) - before, _profile._IDENTITY_CHECK_EVERY)

    def test_waiting_for_room_sleeps_on_the_descriptor(self):
        r, w = os.pipe()
        self.addCleanup(os.close, r)
        self.addCleanup(os.close, w)
        os.set_blocking(w, False)
        fcntl.fcntl(w, fcntl.F_SETPIPE_SZ, 4096)
        while True:
            try:
                os.write(w, b"x" * 4096)
            except BlockingIOError:
                break
        threading.Timer(0.2, os.read, (r, 1 << 16)).start()
        writer = _profile._DropCountWriter(w)
        started = time.monotonic()
        cpu = time.process_time()
        self.assertTrue(writer.write(_profile.encode_end(1)))
        self.assertGreaterEqual(time.monotonic() - started, 0.15)
        self.assertLess(time.process_time() - cpu, 0.1)  # asleep, not spinning

    def test_other_oserror_on_frame_def_is_swallowed_not_dropped_or_raised(self):
        def broken_write(fd, data):
            raise OSError("broken pipe")

        writer = _writer(broken_write)
        emitter = _profile._FrameEmitter(writer.write)
        emitter.emit(_profile.CALL, 1, 10, "f", "/p/a.py", 1)  # must not raise
        self.assertEqual(writer.dropped, 0)  # OSError is swallowed, not counted

    def test_reader_counts_dropped_records_as_lost(self):
        written = []
        emitter = _profile._FrameEmitter(written.append)
        emitter.emit(_profile.CALL, 7, 10, "a", "/p/a.py", 1)
        emitter.emit(_profile.CALL, 7, 20, "b", "/p/a.py", 2)  # DEF b, CALL b
        emitter.emit(_profile.CALL, 7, 30, "c", "/p/a.py", 3)
        # Drop the CALL-b record only (index 3: DEF a, CALL a, DEF b, CALL b,
        # DEF c, CALL c). record_type() returns the wire rtype, which for a
        # CALL frame reference is `_FRAME_CALL_RTYPE`, not the kind constant
        # `CALL` (that constant is 0 and collides with the HELLO rtype).
        kept = [b for i, b in enumerate(written)
                if not (i == 3 and
                        _profile.record_type(b) == _profile._FRAME_CALL_RTYPE)]
        reader = _profile.RecordReader()
        recs = []
        for b in kept:
            recs += reader.feed(b)
        self.assertEqual(reader.lost_records, 1)
        self.assertIn("gap", [r[0] for r in recs])


class ProfileReaderTest(unittest.TestCase):
    def setUp(self):
        _reset_seq()

    def test_poll_flushes_a_pending_gap_left_unresolved_by_this_drain(self):
        # Burst-then-quiet: the gap is first revealed on a first-sighting
        # FRAME_DEF (no ts_ns), and the CALL that would normally resolve it
        # is ALSO dropped under the same overrun. If poll() left the gap
        # pending indefinitely, a pre-gap CALL whose RETURN was dropped
        # could sit in the timeline past Joiner.flush's hold window and get
        # attributed to -- the leaked-frame mis-attribution on_gap exists
        # to prevent. poll() must flush it as ("gap", None) once its own
        # drain empties the queue with no resolving record in hand, not
        # leave it stuck for an indefinite number of future poll() calls.
        r, w = os.pipe()
        os.set_blocking(r, True)
        reader = _profile.ProfileReader(r)
        reader.start()
        os.write(w, _profile.encode_hello(pid=1, start_ns=0))
        os.write(w, _profile._encode_frame_def(0, "a", "/p/a.py", 1))
        _profile._next_seq()  # a real dropped record, never written
        os.write(w, _profile._encode_frame_def(1, "b", "/p/a.py", 2))
        # No later real-ts record follows this drain batch.
        deadline = time.monotonic() + 2.0
        out = []
        while not out and time.monotonic() < deadline:
            out.extend(reader.poll())
            time.sleep(0.005)
        self.assertIn(("gap", None), out)
        os.close(w)
        os.close(r)

    def _poll_until(self, reader, done):
        out = []
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            out.extend(reader.poll())
            if done(out):
                return out
            time.sleep(0.005)
        self.fail(f"poll never reached the expected state: {out}")

    def test_drained_instant_follows_an_empty_stream_only(self):
        r, w = os.pipe()
        self.addCleanup(os.close, r)
        self.addCleanup(os.close, w)
        reader = _profile.ProfileReader(r)
        before_ns = time.monotonic_ns()
        # Not started: the bytes stay in the pipe, so nothing is drained.
        os.write(w, _profile.encode_hello(pid=1, start_ns=0))
        self.assertEqual(reader.poll(), [])
        self.assertEqual(reader.drained_ns, 0)
        reader.start()
        os.write(w, _encode_frame(_profile.CALL, 7, 100, "run", "/p/a.py", 3))
        self._poll_until(reader, lambda out: any(r[0] == "frame" for r in out))
        self._poll_until(reader, lambda out: reader.drained_ns >= before_ns)

    def test_drop_with_no_later_record_is_reported_once_the_stream_is_empty(self):
        # The last records of a burst were dropped and the target then stays
        # inside library code: no later record shows the hole.
        r, w = os.pipe()
        self.addCleanup(os.close, r)
        self.addCleanup(os.close, w)
        drops_fd = os.eventfd(0, os.EFD_NONBLOCK)
        self.addCleanup(os.close, drops_fd)
        reader = _profile.ProfileReader(r, drops_fd=drops_fd)
        reader.start()
        os.write(w, _profile.encode_hello(pid=1, start_ns=0))
        os.write(w, _encode_frame(_profile.CALL, 7, 100, "run", "/p/a.py", 3))
        self._poll_until(reader, lambda out: any(r[0] == "frame" for r in out))
        self.assertNotIn(("gap", None), reader.poll())
        os.eventfd_write(drops_fd, 2)
        self.assertEqual(reader.poll(), [("gap", None)])
        self.assertEqual(reader.poll(), [])  # reported once

    def test_drop_a_later_record_revealed_is_not_reported_again(self):
        r, w = os.pipe()
        self.addCleanup(os.close, r)
        self.addCleanup(os.close, w)
        drops_fd = os.eventfd(0, os.EFD_NONBLOCK)
        self.addCleanup(os.close, drops_fd)
        reader = _profile.ProfileReader(r, drops_fd=drops_fd)
        reader.start()
        os.write(w, _profile.encode_hello(pid=1, start_ns=0))
        os.write(w, _encode_frame(_profile.CALL, 7, 100, "run", "/p/a.py", 3))
        _profile._next_seq()  # one dropped record
        os.eventfd_write(drops_fd, 1)
        os.write(w, _profile._encode_frame_ref(_profile.RETURN, 7, 200, 0))
        out = self._poll_until(
            reader, lambda out: sum(r[0] == "frame" for r in out) == 2)
        out.extend(reader.poll())
        self.assertEqual([r for r in out if r[0] == "gap"], [("gap", 200)])

    def test_poll_decodes_a_bounded_number_of_chunks(self):
        # One call must not keep the event loop from the ring buffer for as
        # long as the target keeps writing.
        r, w = os.pipe()
        self.addCleanup(os.close, r)
        self.addCleanup(os.close, w)
        reader = _profile.ProfileReader(r)
        reader._q.put(_profile.encode_hello(pid=1, start_ns=0))
        for ts in (10, 20, 30):
            reader._q.put(_encode_frame(
                _profile.CALL, 7, ts, "run", "/p/a.py", 3))
        self.assertEqual(reader.poll(max_chunks=2), [
            ("frame", _profile.CALL, 7, 10, "run", "/p/a.py", 3)])
        self.assertTrue(reader.has_backlog())
        self.assertEqual(reader.drained_ns, 0)
        self.assertEqual(len(reader.poll()), 2)
        self.assertFalse(reader.has_backlog())

    def test_backlog_is_bounded_when_the_controller_falls_behind(self):
        r, w = os.pipe()
        reader = _profile.ProfileReader(r, max_chunks=2)
        reader.start()
        os.set_blocking(w, False)
        written = 0
        deadline = time.monotonic() + 5.0
        # With nobody polling, the reader thread must stop taking data, so
        # the pipe fills and the writer sees it is full.
        while time.monotonic() < deadline:
            try:
                written += os.write(w, b"\0" * 65536)
            except BlockingIOError:
                time.sleep(0.05)
                try:
                    os.write(w, b"\0")
                except BlockingIOError:
                    break
                written += 1
        else:
            self.fail("the reader kept draining without bound")
        self.assertLessEqual(reader._q.qsize(), 2)
        os.close(w)
        reader.drain_to_eof(time.monotonic() + 5.0)
        self.assertTrue(reader.at_eof)
        os.close(r)

    def test_reads_records_across_threads_and_reaches_eof(self):
        r, w = os.pipe()
        os.set_blocking(r, True)
        reader = _profile.ProfileReader(r)
        reader.start()
        os.write(w, _profile.encode_hello(pid=1, start_ns=0))
        os.write(w, _encode_frame(_profile.CALL, 7, 100, "run", "/p/a.py", 3))
        deadline = time.monotonic() + 2.0
        got = []
        while not got and time.monotonic() < deadline:
            got.extend(reader.poll())
            time.sleep(0.005)
        self.assertEqual(got, [("frame", _profile.CALL, 7, 100, "run", "/p/a.py", 3)])
        os.write(w, _profile.encode_end(200))
        os.close(w)  # child gone -> EOF
        tail = reader.drain_to_eof(time.monotonic() + 2.0)
        self.assertEqual(tail, [])
        self.assertTrue(reader.at_eof)
        self.assertFalse(reader.ended_early())
        os.close(r)

    def test_eof_without_the_end_record_is_a_cut_short_stream(self):
        # A killed target, or one whose last records were dropped on a full
        # pipe, never writes END. Nothing after the last record is trusted.
        r, w = os.pipe()
        os.set_blocking(r, True)
        reader = _profile.ProfileReader(r)
        reader.start()
        os.write(w, _profile.encode_hello(pid=1, start_ns=0))
        os.write(w, _encode_frame(_profile.CALL, 7, 100, "run", "/p/a.py", 3))
        os.close(w)
        records = reader.drain_to_eof(time.monotonic() + 2.0)
        self.assertEqual(records, [
            ("frame", _profile.CALL, 7, 100, "run", "/p/a.py", 3),
            ("gap", None)])
        self.assertTrue(reader.ended_early())
        self.assertEqual(reader.lost_records(), 1)
        os.close(r)

    def test_a_stream_that_never_started_is_not_cut_short(self):
        # --no-attribution installs no hook, so the pipe carries nothing.
        r, w = os.pipe()
        os.set_blocking(r, True)
        reader = _profile.ProfileReader(r)
        reader.start()
        os.close(w)
        self.assertEqual(reader.drain_to_eof(time.monotonic() + 2.0), [])
        self.assertFalse(reader.ended_early())
        self.assertEqual(reader.lost_records(), 0)
        os.close(r)

    def test_lost_records_passthrough(self):
        # ProfileReader.lost_records() surfaces the inner RecordReader's
        # count -- the authoritative loss number for the summary.
        r, w = os.pipe()
        os.set_blocking(r, True)
        reader = _profile.ProfileReader(r)
        reader.start()
        os.write(w, _profile.encode_hello(pid=1, start_ns=0))
        os.write(w, _profile._encode_frame_def(0, "a", "/p/a.py", 1))
        os.write(w, _profile._encode_frame_ref(_profile.CALL, 7, 10, 0))
        _profile._next_seq()  # consume a seq without writing it: a real drop
        os.write(w, _profile._encode_frame_ref(_profile.CALL, 7, 20, 0))
        deadline = time.monotonic() + 2.0
        got = []
        while len(got) < 2 and time.monotonic() < deadline:
            got.extend(reader.poll())
            time.sleep(0.005)
        os.close(w)
        self.assertEqual(reader.lost_records(), 1)

    def test_drain_to_eof_flushes_pending_gap_on_timeout(self):
        # A grandchild the target forked without exec can hold the write
        # end open past the final-drain deadline, so EOF may never arrive.
        # A gap first revealed on a no-ts record (pending, waiting for a
        # later real-ts record) must still be flushed once the deadline
        # expires -- this is the terminal drain before the joiner's final
        # forced flush, so a pending gap left unflushed here would let
        # that flush attribute against stale pre-gap state.
        r, w = os.pipe()
        os.set_blocking(r, True)
        reader = _profile.ProfileReader(r)
        reader.start()
        os.write(w, _profile.encode_hello(pid=1, start_ns=0))
        os.write(w, _profile._encode_frame_def(0, "a", "/p/a.py", 1))
        _profile._next_seq()  # a real dropped record, never written
        os.write(w, _profile._encode_frame_def(1, "b", "/p/a.py", 2))
        # No later real-ts record follows, and w is never closed.
        out = reader.drain_to_eof(time.monotonic() + 0.2)
        self.assertIn(("gap", None), out)
        self.assertFalse(reader.at_eof)
        os.close(w)
        os.close(r)


class ProjectFileTest(unittest.TestCase):
    def test_inside_root(self):
        self.assertTrue(_is_project_file("/p/x/y.py", "/p"))

    def test_outside_root(self):
        self.assertFalse(_is_project_file("/usr/lib/python3.10/os.py", "/p"))

    def test_site_packages_inside_root_excluded(self):
        self.assertFalse(
            _is_project_file("/p/venv/lib/python3.10/site-packages/m.py", "/p"))

    def test_metagross_itself_excluded(self):
        import metagross
        path = metagross.__file__
        root = os.path.dirname(os.path.dirname(path))
        self.assertFalse(_is_project_file(path, root))

    def test_code_without_a_source_file_is_not_project_code(self):
        # exec(), frozen modules and generated code carry names like
        # "<string>", which resolve to a path under the working directory.
        root = os.getcwd()
        for name in ("<string>", "<stdin>", "<frozen importlib._bootstrap>",
                     "<eval_with_key>.0"):
            self.assertFalse(_is_project_file(name, root), name)

    def test_filesystem_root_accepts_project_file(self):
        self.assertTrue(_is_project_file("/tmp/project.py", "/"))

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
        frames = [rec for rec in records if rec[0] == "frame"]
        funcs = [rec[4] for rec in frames]
        self.assertIn("hot", funcs)
        self.assertNotIn("gap", [rec[0] for rec in records])
        kinds = [rec[1] for rec in frames if rec[4] == "hot"]
        self.assertEqual(sorted(set(kinds)), [_profile.CALL, _profile.RETURN])

    def test_hook_removed_for_new_threads_is_reported_at_exit(self):
        # threading.setprofile() changes the hook later threads start with
        # and raises no audit event, so it is only noticed at exit.
        r, w = os.pipe()
        self.addCleanup(os.close, r)
        code = (
            "import sys, tempfile, threading\n"
            "sys.path.insert(0, %r)\n"
            "from metagross import _profile\n"
            "_profile.install(%d, tempfile.mkdtemp())\n"
            "threading.setprofile(None)\n"
        ) % (os.getcwd(), w)
        try:
            subprocess.run([sys.executable, "-c", code], pass_fds=(w,),
                           check=True, timeout=30)
        finally:
            os.close(w)
        data = b""
        while chunk := os.read(r, 4096):
            data += chunk
        reader = _profile.RecordReader()
        reader.feed(data)
        self.assertEqual(reader.hook_replacements, 1)
        self.assertEqual(reader.end_of_stream(), [])  # END still arrived

    def test_replaced_hook_drops_the_frames_it_can_no_longer_close(self):
        # The script swaps in its own profile function (as cProfile does on
        # Python 3.11 and earlier) while a project frame is open. Nothing
        # after that point may be attributed to the stale frame.
        r, w = os.pipe()
        project = (
            "import sys\n"
            "def before(): pass\n"
            "def after(): pass\n"
            "def outer():\n"
            "    before()\n"
            "    sys.setprofile(lambda *event: None)\n"
            "    after()\n"
            "    sys.setprofile(None)\n"
        )
        code = (
            "import os, sys, tempfile\n"
            "sys.path.insert(0, %r)\n"
            "from metagross import _profile\n"
            "d = tempfile.mkdtemp()\n"
            "with open(os.path.join(d, 'proj.py'), 'w') as stream:\n"
            "    stream.write(%r)\n"
            "sys.path.insert(0, d)\n"
            "_profile.install(%d, d)\n"
            "import proj\n"
            "proj.outer()\n"
        ) % (os.getcwd(), project, w)
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
        reader = _profile.RecordReader()
        records = reader.feed(data)
        self.assertEqual(reader.hook_replacements, 1)
        funcs = [rec[4] for rec in records if rec[0] == "frame"]
        self.assertIn("before", funcs)
        self.assertNotIn("after", funcs)
        gap_ts = [rec[1] for rec in records if rec[0] == "gap"]
        self.assertEqual(len(gap_ts), 1)
        joiner = _events.Joiner()
        for rec in records:
            joiner.on_profile_record(rec)
        tid = next(rec[2] for rec in records if rec[0] == "frame")
        self.assertIsNone(joiner.timeline.attribute(tid, gap_ts[0] + 1))

    def _run_hooked_script(self, project, pipe_size=None):
        """Run `proj.run()` under the real hook in a fresh interpreter.

        Nothing reads the pipe while the script runs. Return its exit status
        and whatever the pipe held once it exited.
        """
        r, w = os.pipe()
        if pipe_size is not None:
            fcntl.fcntl(w, fcntl.F_SETPIPE_SZ, pipe_size)
        code = (
            "import os, sys, tempfile\n"
            "sys.path.insert(0, %r)\n"
            "from metagross import _profile\n"
            "d = tempfile.mkdtemp()\n"
            "with open(os.path.join(d, 'proj.py'), 'w') as stream:\n"
            "    stream.write(%r)\n"
            "sys.path.insert(0, d)\n"
            "_profile.install(%d, d)\n"
            "import proj\n"
            "proj.run()\n"
        ) % (os.getcwd(), project, w)
        process = subprocess.Popen([sys.executable, "-c", code], pass_fds=(w,))
        os.close(w)
        try:
            status = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            status = None
        os.set_blocking(r, False)
        data = b""
        with contextlib.suppress(BlockingIOError):
            while chunk := os.read(r, 65536):
                data += chunk
        os.close(r)
        return status, data

    def test_target_finishes_when_the_controller_stops_reading(self):
        # Nobody reads the pipe, so it fills while the script still has new
        # functions to define. The script must run to its end regardless.
        count = 300
        project = (
            "".join(f"def f{i}(): pass\n" for i in range(count))
            + "def run():\n"
            + "".join(f"    f{i}()\n" for i in range(count))
        )
        status, data = self._run_hooked_script(project, pipe_size=4096)
        self.assertEqual(status, 0, "the script hung on a full profile pipe")
        # Everything after the pipe filled was dropped with no later record
        # to show a seq hole; only the missing END gives the loss away.
        reader = _profile.RecordReader()
        records = reader.feed(data) + reader.end_of_stream()
        self.assertEqual(records[-1], ("gap", None))
        self.assertTrue(reader.ended_early)
        self.assertGreater(reader.lost_records, 0)

    def test_normal_exit_ends_the_stream_once(self):
        # The script forks a child that also exits normally. Only the
        # script itself may end the stream.
        project = (
            "import os, sys\n"
            "def work(): pass\n"
            "def run():\n"
            "    work()\n"
            "    pid = os.fork()\n"
            "    if pid == 0:\n"
            "        sys.exit(0)\n"
            "    os.waitpid(pid, 0)\n"
            "    work()\n"
        )
        status, data = self._run_hooked_script(project)
        self.assertEqual(status, 0)
        reader = _profile.RecordReader()
        records = reader.feed(data) + reader.end_of_stream()
        self.assertNotIn("gap", [rec[0] for rec in records])
        self.assertFalse(reader.ended_early)
        self.assertEqual(reader.lost_records, 0)
        self.assertEqual(
            [rec[4] for rec in records if rec[0] == "frame"].count("work"), 4)

    def test_killed_script_leaves_a_cut_short_stream(self):
        project = (
            "import os, signal\n"
            "def run():\n"
            "    os.kill(os.getpid(), signal.SIGKILL)\n"
        )
        status, data = self._run_hooked_script(project)
        self.assertEqual(status, -signal.SIGKILL)
        reader = _profile.RecordReader()
        records = reader.feed(data) + reader.end_of_stream()
        self.assertEqual(records[-1], ("gap", None))
        self.assertTrue(reader.ended_early)

    def test_asyncio_tasks_keep_their_own_spans(self):
        # Three tasks interleave on one thread: two hold their own span
        # across awaits and one has none. Each probe call must see the span
        # of the task that made it.
        r, w = os.pipe()
        project = (
            "import asyncio\n"
            "import metagross\n"
            "def probe_a(): pass\n"
            "def probe_b(): pass\n"
            "def probe_c(): pass\n"
            "async def spanned(name, probe):\n"
            "    with metagross.span(name):\n"
            "        for _ in range(3):\n"
            "            await asyncio.sleep(0)\n"
            "            probe()\n"
            "async def bare(probe):\n"
            "    for _ in range(3):\n"
            "        await asyncio.sleep(0)\n"
            "        probe()\n"
            "async def main():\n"
            "    await asyncio.gather(spanned('request-a', probe_a),\n"
            "                         spanned('request-b', probe_b),\n"
            "                         bare(probe_c))\n"
            "def run():\n"
            "    asyncio.run(main())\n"
        )
        code = (
            "import os, sys, tempfile\n"
            "sys.path.insert(0, %r)\n"
            "from metagross import _profile\n"
            "d = tempfile.mkdtemp()\n"
            "with open(os.path.join(d, 'proj.py'), 'w') as stream:\n"
            "    stream.write(%r)\n"
            "sys.path.insert(0, d)\n"
            "_profile.install(%d, d)\n"
            "import proj\n"
            "proj.run()\n"
        ) % (os.getcwd(), project, w)
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
        self.assertNotIn("gap", [rec[0] for rec in records])
        joiner = _events.Joiner()
        for rec in records:
            joiner.on_profile_record(rec)
        seen = {}
        for rec in records:
            if rec[0] == "frame" and rec[1] == _profile.CALL:
                _, _, tid, ts, func, _, _ = rec
                if func.startswith("probe_"):
                    seen.setdefault(func, []).append(
                        joiner.spans.attribute(tid, ts))
        self.assertEqual(seen, {
            "probe_a": ["request-a"] * 3,
            "probe_b": ["request-b"] * 3,
            "probe_c": [None] * 3,
        })

    def test_forked_child_stops_profiling_and_does_not_corrupt_seq(self):
        # A target that forks without exec (multiprocessing/DataLoader
        # workers) inherits the hook, the pipe fd, and the seq counter's
        # current value. If the child kept tracing, its records would
        # interleave into the parent's seq stream and manufacture false
        # gaps. install() must disable tracing in the forked child.
        r, w = os.pipe()
        code = (
            "import os, sys, tempfile\n"
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
            "child_pid = os.fork()\n"
            "if child_pid == 0:\n"
            "    proj.hot()\n"
            "    os._exit(0)\n"
            "os.waitpid(child_pid, 0)\n"
            "proj.hot()\n"
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
        self.assertNotIn("gap", [rec[0] for rec in records])


class SpanApiTest(unittest.TestCase):
    def test_span_is_noop_without_profiling_installed(self):
        # Outside a metagross run, span() must not raise or write anywhere.
        with metagross.span("anything"):
            pass  # no profiling handle installed -> no-op

    def _capture_span_reports(self):
        """Install a recording emitter; return the decoded span names."""
        _reset_seq()
        written = []
        self.addCleanup(setattr, _profile, "_current_emitter",
                        _profile._current_emitter)
        self.addCleanup(vars(_profile._reported_span).clear)
        vars(_profile._reported_span).clear()
        _profile._current_emitter = _profile._FrameEmitter(written.append)
        hello = _profile.encode_hello(pid=1, start_ns=0)

        def names():
            reader = _profile.RecordReader()
            records = reader.feed(hello + b"".join(written))
            self.assertNotIn("gap", [rec[0] for rec in records])
            return [rec[3] for rec in records if rec[0] == "span"]

        return names

    def test_nested_spans_report_the_innermost_open_span(self):
        names = self._capture_span_reports()
        with metagross.span("forward"):
            with metagross.span("matmul"):
                pass
        self.assertEqual(names(), ["forward", "matmul", "forward", None])

    def test_span_closed_out_of_order_reports_none_until_unwound(self):
        # Two generators interleave in one context, so the first span closes
        # while the second is innermost. Neither can be trusted afterwards.
        names = self._capture_span_reports()

        def holder(name):
            with metagross.span(name):
                yield

        first, second = holder("first"), holder("second")
        next(first)
        next(second)
        first.close()
        second.close()
        with metagross.span("later"):
            pass
        self.assertEqual(names(), ["first", "second", None, "later", None])

    def test_inherited_span_stops_reporting_once_it_closes(self):
        names = self._capture_span_reports()
        with metagross.span("request"):
            inherited = contextvars.copy_context()
        # A task or thread started inside the span still holds it, but the
        # span is closed: it must not label that work.
        inherited.run(_profile.report_span)
        self.assertEqual(names(), ["request", None])

    def test_span_closed_from_another_context_leaves_that_context_alone(self):
        names = self._capture_span_reports()
        other = contextvars.copy_context()
        stranded = metagross.span("stranded")
        other.run(stranded.__enter__)
        with metagross.span("mine"):
            stranded.__exit__(None, None, None)
            _profile.report_span()
            self.assertEqual(names(), ["stranded", "mine"])
        other.run(_profile.report_span)
        self.assertEqual(names(), ["stranded", "mine", None])


class OpSpanTimelineTest(unittest.TestCase):
    def test_active_span_is_the_last_report_at_or_before_ts(self):
        t = _events.OpSpanTimeline()
        t.on_span(7, 10, "forward")
        t.on_span(7, 20, "matmul")
        t.on_span(7, 30, "forward")
        t.on_span(7, 40, None)
        self.assertIsNone(t.attribute(7, 5))
        self.assertEqual(t.attribute(7, 25), "matmul")
        self.assertEqual(t.attribute(7, 35), "forward")
        self.assertIsNone(t.attribute(7, 45))
        self.assertIsNone(t.attribute(8, 25))

    def test_gap_fails_closed_until_the_next_report(self):
        t = _events.OpSpanTimeline()
        t.on_span(7, 10, "forward")
        t.on_gap(50)
        self.assertIsNone(t.attribute(7, 20))
        self.assertIsNone(t.attribute(7, 60))
        t.on_span(7, 70, "matmul")
        self.assertEqual(t.attribute(7, 80), "matmul")

    def test_prune_keeps_the_span_active_at_the_horizon(self):
        t = _events.OpSpanTimeline()
        t.on_span(7, 10, "warmup")
        t.on_span(7, 20, "forward")
        t.prune(100)
        self.assertEqual(t._logs[7], [(20, "forward")])
        self.assertEqual(t.attribute(7, 150), "forward")
        self.assertIsNone(t.attribute(7, 50))

    def test_prune_drops_a_closed_history(self):
        t = _events.OpSpanTimeline()
        t.on_span(7, 10, "forward")
        t.on_span(7, 20, None)
        t.prune(100)
        self.assertNotIn(7, t._logs)


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

    def test_mismatched_return_does_not_pop_a_live_frame(self):
        tl = _events.FrameTimeline()
        tl.on_record(0, 7, 10, "outer", "/proj/a.py", 1)   # call outer
        tl.on_record(0, 7, 20, "inner", "/proj/b.py", 2)   # call inner
        tl.on_record(1, 7, 30, "ghost", "/proj/c.py", 3)   # spurious return
        # inner is still open; the spurious return must not unwind it.
        self.assertEqual(tl.attribute(7, 35),
                         _events.FrameInfo("inner", "/proj/b.py", 2))

    def test_prune_bounds_history_under_long_lived_frame(self):
        tl = _events.FrameTimeline()
        tl.on_record(0, 3, 1, "main", "/proj/m.py", 1)   # long-lived outer frame
        for k in range(50):
            ts = 2 + 2 * k
            tl.on_record(0, 3, ts, "step", "/proj/m.py", 9)
            tl.on_record(1, 3, ts + 1, "step", "/proj/m.py", 9)
        tl.attribute(3, 200)                              # advance the cursor
        tl.prune(150)
        log = tl._logs[3]
        self.assertLessEqual(len(log), 2)                 # main + nothing still open
        # Attribution after prune is still correct.
        self.assertEqual(tl.attribute(3, 300),
                         _events.FrameInfo("main", "/proj/m.py", 1))

    def test_prune_horizon_fails_closed_on_pruned_history(self):
        tl = _events.FrameTimeline()
        tl.on_record(0, 1, 1, "main", "/p/m.py", 1)     # main opens
        tl.on_record(0, 1, 10, "inner", "/p/m.py", 5)   # inner opens
        tl.on_record(1, 1, 20, "inner", "/p/m.py", 5)   # inner closes
        tl.attribute(1, 100)                            # advance the cursor
        tl.prune(50)                                    # inner's open/close is pruned
        # The true answer at ts=15 was inner, now pruned away; must be None,
        # not a confident-wrong "main".
        self.assertIsNone(tl.attribute(1, 15))
        self.assertEqual(tl.attribute(1, 100),
                         _events.FrameInfo("main", "/p/m.py", 1))


    def test_a_query_below_the_horizon_is_counted_as_refused(self):
        tl = _events.FrameTimeline()
        tl.on_record(0, 1, 10, "f", "/p/a.py", 1)
        self.assertIsNone(tl.attribute(2, 50))   # a thread with no frames
        self.assertEqual(tl.refused, 0)
        tl.on_gap(100)
        self.assertIsNone(tl.attribute(1, 50))   # its history was dropped
        self.assertEqual(tl.refused, 1)


class GapHandlingTest(unittest.TestCase):
    def setUp(self):
        _reset_seq()

    def test_seq_gap_emits_gap_record(self):
        reader = _profile.RecordReader()
        out = reader.feed(_profile.encode_hello(pid=1234, start_ns=1))
        out += reader.feed(_encode_frame(_profile.CALL, tid=7, ts_ns=10,
                                                 func="a", path="/p/a.py", line=1))
        # A real dropped record: encode one (it consumes a seq) but never feed
        # it, so the reader sees the seq jump.
        _encode_frame(_profile.CALL, tid=7, ts_ns=20, func="x",
                              path="/p/a.py", line=9)
        out += reader.feed(_encode_frame(_profile.CALL, tid=7, ts_ns=30,
                                                 func="b", path="/p/a.py", line=2))
        self.assertIn("gap", [rec[0] for rec in out])
        # The gap tuple precedes the record that revealed it, so a consumer
        # can prune/reset before trusting the record that follows.
        gap_index = [rec[0] for rec in out].index("gap")
        self.assertEqual(out[gap_index + 1][0], "frame")

    def test_gap_first_revealed_on_frame_def_carries_next_real_ts(self):
        # A gap revealed on a FRAME_DEF (no ts_ns of its own) must not be
        # reported with an unknown ts: it is held pending and reported with
        # the ts of the next record that DOES carry a real one (the tighter
        # horizon), not left for the Joiner's much wider
        # "now at decode time" monotonic fallback.
        reader = _profile.RecordReader()
        out = reader.feed(_profile.encode_hello(pid=1, start_ns=1))
        out += reader.feed(_profile._encode_frame_def(0, "a", "/p/a.py", 1))
        _profile._next_seq()  # a real dropped record: consumes a seq, never fed
        # The gap is first revealed here, on a FRAME_DEF -- no ts_ns yet.
        out += reader.feed(_profile._encode_frame_def(1, "b", "/p/a.py", 2))
        self.assertNotIn("gap", [rec[0] for rec in out])  # deferred, not fired
        out += reader.feed(_profile._encode_frame_ref(_profile.CALL, 7, 42, 1))
        gap_recs = [rec for rec in out if rec[0] == "gap"]
        self.assertEqual(gap_recs, [("gap", 42)])  # the CALL's real ts, not None

    def test_gap_pending_on_no_ts_record_flushes_as_none_at_eof(self):
        # If the stream ends before a real-ts record ever follows the
        # no-ts record that first revealed the gap, the pending gap must
        # still surface -- as the ("gap", None) shape the Joiner already
        # falls back to its own monotonic clock for -- rather than
        # silently vanishing.
        reader = _profile.RecordReader()
        reader.feed(_profile.encode_hello(pid=1, start_ns=1))
        reader.feed(_profile._encode_frame_def(0, "a", "/p/a.py", 1))
        _profile._next_seq()  # a real dropped record, never fed
        out = reader.feed(_profile._encode_frame_def(1, "b", "/p/a.py", 2))
        self.assertEqual(out, [])  # deferred, not fired yet
        self.assertEqual(reader.finalize(), [("gap", None)])
        self.assertEqual(reader.finalize(), [])  # nothing left to flush twice

    def test_frame_timeline_gap_fails_closed(self):
        tl = _events.FrameTimeline()
        tl.on_record(0, 7, 10, "a", "/p/a.py", 1)
        tl.attribute(7, 10)
        tl.on_gap(50)
        self.assertIsNone(tl.attribute(7, 20))   # below the gap horizon -> unknown

    def test_gap_after_lost_return_does_not_leave_a_stale_open_frame(self):
        # A CALL is recorded and its replay cursor is advanced past it (so
        # a bare state reset alone, without also dropping the record, would
        # let it replay again on the forward branch); its matching RETURN
        # is then lost to the gap. on_gap must drop the CALL along with
        # everything else before the gap, so queries both below the
        # horizon and at/after it fail closed rather than reporting a
        # confidently wrong frame.
        tl = _events.FrameTimeline()
        tl.on_record(0, 1, 10, "leaked", "/p/a.py", 1)   # CALL; its RETURN is lost
        tl.attribute(1, 10)                              # advance the replay cursor
        tl.on_gap(100)
        self.assertIsNone(tl.attribute(1, 50))            # below horizon -> unknown
        self.assertIsNone(tl.attribute(1, 200))           # at/after gap -> still unknown

    def test_on_gap_drops_all_pre_gap_records_including_open_calls(self):
        # on_gap must drop everything before the gap -- closed pairs AND
        # still-open CALLs alike -- not retain open CALLs the way prune()
        # does for an ordinary (non-gap) prune. See
        # test_gap_does_not_confidently_attribute_to_a_leaked_open_call for
        # why retaining an open CALL across a gap is itself a bug.
        tl = _events.FrameTimeline()
        tl.on_record(0, 1, 10, "done", "/p/a.py", 1)     # CALL, closed pair
        tl.on_record(1, 1, 20, "done", "/p/a.py", 1)     # RETURN
        tl.on_record(0, 1, 30, "leaked", "/p/a.py", 2)   # CALL; its RETURN is lost
        tl.attribute(1, 30)
        tl.on_gap(50)
        self.assertNotIn(1, tl._logs)                    # nothing pre-gap survives
        self.assertIsNone(tl.attribute(1, 40))            # below horizon -> unknown
        self.assertIsNone(tl.attribute(1, 100))           # at/after gap -> still unknown

    def test_gap_does_not_confidently_attribute_to_a_leaked_open_call(self):
        # The regression this fix closes: a CALL's matching RETURN is
        # exactly the record the gap dropped, so it looks indistinguishable
        # from a genuinely still-open CALL. Retaining it (the old
        # prune-and-retain shape) would let every top-level-idle query
        # at/after the gap confidently -- and wrongly -- attribute to it
        # forever. That violates "never a guessed frame."
        tl = _events.FrameTimeline()
        tl.on_record(0, 7, 10, "foo", "/p/a.py", 1)   # CALL foo; its RETURN is lost
        tl.on_gap(50)
        self.assertIsNone(tl.attribute(7, 100))       # NOT "foo"

    def test_gap_horizon_survives_later_prune_with_lower_ts(self):
        # prune() is also called from Joiner.flush() with whatever ts the
        # oldest still-held GPU event has, independent of any gap. That
        # value can be lower than a horizon a gap already raised; the
        # horizon must not retreat just because of the call order.
        tl = _events.FrameTimeline()
        tl.on_record(0, 1, 10, "outer", "/p/a.py", 1)
        tl.attribute(1, 10)
        tl.on_gap(500)
        tl.prune(50)
        # A late record from before the gap gives the thread a log again; the
        # horizon alone must keep that history unanswerable.
        tl.on_record(0, 1, 60, "stale", "/p/a.py", 1)
        self.assertIsNone(tl.attribute(1, 100))

    def test_joiner_dispatches_gap_to_timeline(self):
        j = _events.Joiner(hold_ns=100)
        j.on_profile_record(("frame", _profile.CALL, 1, 10, "f", "/p/a.py", 1))
        j.timeline.attribute(1, 10)
        j.on_profile_record(("gap", 500))
        self.assertIsNone(j.timeline.attribute(1, 20))

    def test_joiner_gap_without_ts_uses_monotonic_clock(self):
        j = _events.Joiner(hold_ns=100)
        j.on_profile_record(("frame", _profile.CALL, 1, 10, "f", "/p/a.py", 1))
        j.on_profile_record(("gap", None))
        # The real horizon is "now" (a huge monotonic_ns value); a query at
        # the tiny fabricated ts=10 must fail closed either way.
        self.assertIsNone(j.timeline.attribute(1, 10))


class JoinerTest(unittest.TestCase):
    def test_span_reports_reach_the_span_timeline(self):
        j = _events.Joiner()
        j.on_profile_record(("span", 7, 10, "forward"))
        j.on_profile_record(("span", 7, 20, "matmul"))
        j.on_profile_record(("span", 7, 30, "forward"))
        j.on_profile_record(("span", 7, 40, None))
        self.assertEqual(j.spans.attribute(7, 25), "matmul")
        self.assertEqual(j.spans.attribute(7, 35), "forward")
        self.assertIsNone(j.spans.attribute(7, 45))

    def test_hold_then_release(self):
        j = _events.Joiner(hold_ns=100)
        j.on_profile_record(("frame", _profile.CALL, 1, 10, "f", "/p/a.py", 1))
        j.on_gpu_event(_raw(15, args=(0x77,), ts=50, dur=5, tid=1))
        self.assertEqual(j.flush(now_ns=100), [])          # still held
        released = j.flush(now_ns=200)
        self.assertEqual(len(released), 1)
        self.assertEqual(released[0].frame.function, "f")
        self.assertEqual(released[0].api.base, "cuStreamSynchronize")

    def test_late_profile_record_beats_hold(self):
        j = _events.Joiner(hold_ns=100)
        j.on_gpu_event(_raw(16, ts=50, dur=5, tid=1))
        j.on_profile_record(("frame", _profile.CALL, 1, 10, "f", "/p/a.py", 1))
        self.assertEqual(j.flush(now_ns=200)[0].frame.function, "f")

    def test_long_call_keeps_frame_and_span_after_an_earlier_release(self):
        # A call that blocks past the hold window is delivered at return but
        # stamped with its entry time. Releasing an earlier event must not
        # put the frame it entered from out of reach.
        j = _events.Joiner(hold_ns=100)
        j.on_profile_record(("frame", _profile.CALL, 1, 10, "step", "/p/a.py", 1))
        j.on_profile_record(("span", 1, 12, "epoch"))
        j.on_gpu_event(_raw(16, ts=20, dur=1, tid=1))
        self.assertEqual(len(j.flush(now_ns=200)), 1)
        j.on_gpu_event(_raw(15, ts=30, dur=470, tid=1))    # returned at 500
        out = j.flush(now_ns=520)
        self.assertEqual(out[0].frame.function, "step")
        self.assertEqual(out[0].span, "epoch")

    def test_long_call_is_attributed_to_the_frame_it_blocked_in(self):
        # Another thread's releases prune while tid 1 is blocked, and tid 1
        # moves on to a new frame before its own event is delivered.
        j = _events.Joiner(hold_ns=100)
        j.on_profile_record(("frame", _profile.CALL, 1, 10, "wait", "/p/a.py", 1))
        j.on_profile_record(("frame", _profile.CALL, 2, 40, "other", "/p/a.py", 9))
        j.on_gpu_event(_raw(16, ts=50, dur=1, tid=2))
        self.assertEqual(len(j.flush(now_ns=200)), 1)
        j.on_gpu_event(_raw(16, ts=300, dur=1, tid=2))
        self.assertEqual(len(j.flush(now_ns=450)), 1)
        j.on_profile_record(("frame", _profile.RETURN, 1, 505, "wait", "/p/a.py", 1))
        j.on_profile_record(("frame", _profile.CALL, 1, 506, "next", "/p/a.py", 5))
        j.on_gpu_event(_raw(15, ts=30, dur=470, tid=1))    # returned at 500
        out = j.flush(now_ns=520)
        self.assertEqual(out[0].frame.function, "wait")

    def test_event_delivered_after_a_release_keeps_its_frame(self):
        # An event that returned just before a flush but reaches the
        # controller after it is still inside the delivery budget.
        j = _events.Joiner(hold_ns=100)
        j.on_profile_record(("frame", _profile.CALL, 1, 10, "f", "/p/a.py", 1))
        j.on_gpu_event(_raw(16, ts=20, dur=1, tid=1))
        self.assertEqual(len(j.flush(now_ns=1000)), 1)
        j.on_gpu_event(_raw(16, ts=990, dur=2, tid=1))
        self.assertEqual(j.flush(now_ns=1100)[0].frame.function, "f")

    def test_call_delivered_after_a_slow_profile_drain_keeps_its_frame(self):
        # The ring buffer is polled, then the profile drain takes 150 ms.
        # A call that returned during the drain is delivered on the next
        # tick; history must reach back to the poll, not to "now".
        ms = 1_000_000
        joiner = _events.Joiner()
        joiner.on_profile_record(
            ("frame", _profile.CALL, 1, 10 * ms, "train_step", "/p/train.py", 12))
        self.assertEqual(
            joiner.flush(160 * ms, delivered_until_ns=10 * ms), [])
        joiner.on_gpu_event(_raw(1, ts=15 * ms, dur=1 * ms, tid=1))
        (event,) = joiner.flush(170 * ms, delivered_until_ns=160 * ms)
        self.assertEqual(event.frame.function, "train_step")

    def test_call_inside_a_profile_hole_waits_and_is_not_guessed(self):
        # train_epoch returns and evaluate is entered, but both records are
        # dropped on a full pipe. evaluate then launches a kernel. The hole
        # shows only when the next record gets through.
        ms = 1_000_000
        _reset_seq()
        reader = _profile.RecordReader()
        joiner = _events.Joiner()

        def feed(data):
            for rec in reader.feed(data):
                joiner.on_profile_record(rec)

        feed(_profile.encode_hello(pid=1, start_ns=0))
        feed(_profile._encode_frame_def(0, "train_epoch", "/p/train.py", 1))
        feed(_profile._encode_frame_def(1, "evaluate", "/p/train.py", 9))
        feed(_profile._encode_frame_ref(_profile.CALL, 1, 10 * ms, 0))
        _profile._encode_frame_ref(_profile.RETURN, 1, 11 * ms, 0)   # dropped
        _profile._encode_frame_ref(_profile.CALL, 1, 12 * ms, 1)     # dropped
        joiner.on_gpu_event(_raw(1, ts=13 * ms, dur=1 * ms, tid=1))
        # The pipe is still full, so the stream has not been read past the
        # call: releasing now would name train_epoch.
        self.assertEqual(
            joiner.flush(120 * ms, profile_drained_ns=5 * ms), [])
        feed(_profile._encode_frame_ref(_profile.RETURN, 1, 170 * ms, 1))
        (event,) = joiner.flush(180 * ms, profile_drained_ns=5 * ms)
        self.assertIsNone(event.frame)
        self.assertEqual(reader.lost_records, 2)
        self.assertEqual(joiner.refused_attributions, 1)

    def test_long_call_waits_for_the_profile_stream_too(self):
        # A call longer than the hold window is delivered with its hold
        # already spent. It still waits for the stream to reach its entry.
        ms = 1_000_000
        joiner = _events.Joiner()
        joiner.on_profile_record(
            ("frame", _profile.CALL, 1, 10 * ms, "load_batch", "/p/a.py", 1))
        joiner.on_gpu_event(_raw(15, ts=20 * ms, dur=300 * ms, tid=1))
        self.assertEqual(
            joiner.flush(321 * ms, profile_drained_ns=15 * ms), [])
        joiner.on_profile_record(("gap", 322 * ms))
        (event,) = joiner.flush(323 * ms, profile_drained_ns=15 * ms)
        self.assertIsNone(event.frame)

    def test_call_from_a_quiet_script_is_released_once_the_stream_is_empty(self):
        # Library code makes the calls and writes no profile records. An
        # empty stream after the call's entry is as good as a newer record.
        ms = 1_000_000
        joiner = _events.Joiner()
        joiner.on_profile_record(
            ("frame", _profile.CALL, 1, 10 * ms, "main", "/p/a.py", 1))
        joiner.on_gpu_event(_raw(1, ts=500 * ms, dur=1 * ms, tid=1))
        (event,) = joiner.flush(700 * ms, profile_drained_ns=650 * ms)
        self.assertEqual(event.frame.function, "main")
        self.assertEqual(joiner.refused_attributions, 0)

    def test_call_the_profile_stream_never_reaches_is_released_unknown(self):
        ms = 1_000_000
        joiner = _events.Joiner()
        joiner.on_profile_record(
            ("frame", _profile.CALL, 1, 10 * ms, "main", "/p/a.py", 1))
        joiner.on_gpu_event(_raw(1, ts=500 * ms, dur=1 * ms, tid=1))
        self.assertEqual(
            joiner.flush(5400 * ms, profile_drained_ns=20 * ms), [])
        (event,) = joiner.flush(5600 * ms, profile_drained_ns=20 * ms)
        self.assertIsNone(event.frame)
        self.assertEqual(joiner.refused_attributions, 1)

    def test_waiting_calls_are_bounded_in_number(self):
        # A target that keeps the profile stream behind must not make the
        # tracer hold calls without limit. The oldest are given up first.
        joiner = _events.Joiner(max_pending=2)
        for ts in (100, 200, 300):
            joiner.on_gpu_event(_raw(16, ts=ts, dur=1, tid=1))
        (event,) = joiner.flush(400, profile_drained_ns=50)
        self.assertEqual(event.raw.ts, 100)
        self.assertIsNone(event.frame)
        self.assertEqual(len(joiner._pending), 2)
        self.assertEqual(joiner.refused_attributions, 1)

    def test_call_held_for_the_profile_stream_keeps_its_history(self):
        # Ticks pass while the call waits; pruning must not put the frame it
        # was made from out of reach.
        ms = 1_000_000
        joiner = _events.Joiner()
        joiner.on_profile_record(
            ("frame", _profile.CALL, 1, 10 * ms, "outer", "/p/a.py", 1))
        joiner.on_profile_record(
            ("frame", _profile.CALL, 1, 20 * ms, "inner", "/p/a.py", 5))
        joiner.on_gpu_event(_raw(1, ts=30 * ms, dur=1 * ms, tid=1))
        for now in (200, 400, 600):
            self.assertEqual(
                joiner.flush(now * ms, profile_drained_ns=25 * ms), [])
        joiner.on_profile_record(
            ("frame", _profile.RETURN, 1, 610 * ms, "inner", "/p/a.py", 5))
        (event,) = joiner.flush(620 * ms, profile_drained_ns=25 * ms)
        self.assertEqual(event.frame.function, "inner")
        self.assertEqual(joiner.refused_attributions, 0)

    def test_gap_reaches_past_every_record_already_seen(self):
        # Another thread read the clock, lost the processor, and wrote the
        # record that reveals the hole with a timestamp older than a CALL
        # written before the hole. That CALL must not survive the gap.
        joiner = _events.Joiner(hold_ns=100)
        joiner.on_profile_record(("frame", _profile.CALL, 1, 500, "helper", "/p/a.py", 1))
        joiner.on_profile_record(("gap", 400))
        joiner.on_profile_record(("frame", _profile.CALL, 2, 400, "other", "/p/a.py", 9))
        self.assertIsNone(joiner.timeline.attribute(1, 600))

    def test_finished_threads_leave_no_state_behind(self):
        # One thread per request: each runs a project function inside a span
        # and exits. Nothing may be kept per thread once its history is old.
        j = _events.Joiner(hold_ns=100)
        for tid in range(1, 1001):
            ts = tid * 10
            j.on_profile_record(("span", tid, ts, "request"))
            j.on_profile_record(("frame", _profile.CALL, tid, ts + 1, "f", "/p/a.py", 1))
            j.on_profile_record(("frame", _profile.RETURN, tid, ts + 2, "f", "/p/a.py", 1))
            j.on_profile_record(("span", tid, ts + 3, None))
        j.flush(now_ns=20_000)
        self.assertEqual(j.timeline._logs, {})
        self.assertEqual(j.timeline._states, {})
        self.assertEqual(j.spans._logs, {})

    def test_flush_still_prunes_history_older_than_the_hold_window(self):
        j = _events.Joiner(hold_ns=100)
        for ts in range(10, 400, 10):
            j.on_profile_record(("frame", _profile.CALL, 1, ts, "f", "/p/a.py", 1))
            j.on_profile_record(("frame", _profile.RETURN, 1, ts + 5, "f", "/p/a.py", 1))
        j.on_gpu_event(_raw(16, ts=12, dur=1, tid=1))
        j.flush(now_ns=1000)
        self.assertNotIn(1, j.timeline._logs)

    def test_flush_prunes_history_while_the_gpu_is_idle(self):
        # No GPU event is ever released here; the frame log must still not
        # grow with every Python call the target makes.
        j = _events.Joiner(hold_ns=100)
        for ts in range(10, 400, 10):
            j.on_profile_record(("frame", _profile.CALL, 1, ts, "f", "/p/a.py", 1))
            j.on_profile_record(("frame", _profile.RETURN, 1, ts + 5, "f", "/p/a.py", 1))
        self.assertEqual(j.flush(now_ns=1000), [])
        self.assertNotIn(1, j.timeline._logs)

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

    def test_launch_keeps_kernel_name_from_enqueue_time(self):
        j = _events.Joiner(hold_ns=100)
        j.on_gpu_event(_raw(18, out=0xF00, name=b"vec_add"))   # register F00 -> vec_add
        j.on_gpu_event(_raw(1, ts=10, args=(0xF00, 1, 1, 1, 1, 1, 1, 0, 0)))  # launch F00
        j.on_gpu_event(_raw(18, out=0xF00, name=b"other"))     # reused handle re-registered
        out = j.flush(10_000, force=True)
        enriched = [j.enrich(e) for e in out]
        self.assertEqual(enriched[0].kernel, "vec_add")

    def test_launch_of_unregistered_handle_freezes_to_placeholder(self):
        j = _events.Joiner(hold_ns=100)
        j.on_gpu_event(_raw(1, ts=10, args=(0xF00, 1, 1, 1, 1, 1, 1, 0, 0)))  # launch F00, unregistered
        j.on_gpu_event(_raw(18, out=0xF00, name=b"late"))  # registered after enqueue
        out = j.flush(10_000, force=True)
        enriched = [j.enrich(e) for e in out]
        self.assertEqual(enriched[0].kernel, "kernel@0xf00")


class SpanJoinTest(unittest.TestCase):
    def test_event_resolves_enclosing_span(self):
        j = _events.Joiner(hold_ns=100)
        j.on_profile_record(("span", 1, 5, "forward"))
        # a launch at ts=10, tid=1, inside the "forward" span
        j.on_gpu_event(_raw(1, ts=10, tid=1,
                            args=(0xF00, 1, 1, 1, 1, 1, 1, 0, 0)))
        out = j.flush(10_000, force=True)
        self.assertEqual(out[0].span, "forward")
        enriched = j.enrich(out[0])
        rec = _events.event_record(enriched, 0, 4242)
        self.assertEqual(rec["span"], "forward")

    def test_event_with_no_enclosing_span_is_null(self):
        j = _events.Joiner(hold_ns=100)
        j.on_gpu_event(_raw(1, ts=10, tid=1,
                            args=(0xF00, 1, 1, 1, 1, 1, 1, 0, 0)))
        out = j.flush(10_000, force=True)
        self.assertIsNone(out[0].span)
        enriched = j.enrich(out[0])
        self.assertIsNone(_events.event_record(enriched, 0, 4242)["span"])

    def test_flush_prunes_span_timeline_with_same_horizon_as_frame_timeline(self):
        j = _events.Joiner(hold_ns=100)
        j.on_profile_record(("frame", _profile.CALL, 1, 4, "f", "/p/a.py", 1))
        j.on_profile_record(("span", 1, 5, "forward"))
        j.on_profile_record(("span", 1, 8, None))
        j.on_gpu_event(_raw(1, ts=10, tid=1,
                            args=(0xF00, 1, 1, 1, 1, 1, 1, 0, 0)))
        j.flush(10_000, force=True)
        # The closed span log is fully below the flush horizon and must be
        # pruned away just like FrameTimeline -- the span timeline must not
        # grow unbounded.
        self.assertEqual(j.spans._logs.get(1, []), [])
        self.assertEqual(j.timeline._horizon, j.spans._horizon)


class ExampleCaptureTest(unittest.TestCase):
    def test_example_capture_has_the_current_record_and_summary_shape(self):
        # The reference points readers at these files as complete examples.
        raw = _raw(16, ts=1, dur=1, tid=1)
        event = _events.EnrichedEvent(raw, _bpf.API_BY_ID[16], None, None, {})
        record_keys = set(_events.event_record(event, 0, 1))
        with open("examples/captures/basic.jsonl", encoding="utf-8") as stream:
            for line in stream:
                self.assertEqual(set(json.loads(line)), record_keys)
        snapshot = _events.CaptureStats().snapshot(
            lost_events=0, dropped_nested_calls=0,
            observed_outstanding_bytes=0, render_failed=False,
            trace_failed=False)
        with open("examples/captures/basic-summary.json",
                  encoding="utf-8") as stream:
            summary = json.load(stream)
        self.assertEqual(set(summary) - {"configuration", "target"},
                         set(snapshot))
        self.assertEqual(set(summary["capture"]), set(snapshot["capture"]))


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
                                     _events.FrameInfo("train_step", "/p/train.py", 31),
                                     kernel_at_enqueue="vec_add")
        out = self._emit(ev, json_output=False)
        self.assertIn("LaunchKernel", out)
        self.assertIn("train_step", out)
        self.assertIn("train.py:31", out)
        self.assertIn("kernel=vec_add", out)
        self.assertIn("20.0us", out)

    def test_table_columns_fit_every_api_and_short_durations(self):
        # Most driver calls take a few microseconds, which two decimals of
        # a millisecond showed as 0.00ms.
        header = None
        for api in _bpf.APIS:
            if api.category == "register":
                continue
            raw = _raw(api.api_id, ts=3_600_000_000_000, dur=3_500, tid=1)
            out = self._emit(_events.AttributedEvent(raw, api, None),
                             json_output=False)
            header, row = out.splitlines()
            with self.subTest(api=api.base):
                self.assertEqual(row[header.index("RET"):][:1], "0")
                self.assertEqual(
                    row[header.index("DURATION"):].split()[0], "3.5us")

    def test_table_cuts_a_long_kernel_name_but_json_keeps_it(self):
        name = "_ZN2at6native" + "x" * 600
        raw = _raw(1, args=(0xF00, 256, 1, 1, 128, 1, 1, 0, 0x77),
                   ts=3_600_000_000_000, dur=20_000, tid=1)
        ev = _events.AttributedEvent(raw, _bpf.API_BY_ID[1],
                                     _events.FrameInfo("train_step", "/p/train.py", 31),
                                     kernel_at_enqueue=name)
        table = self._emit(ev, json_output=False)
        self.assertIn("kernel=" + name[:124] + "... grid=", table)
        record = json.loads(self._emit(ev, json_output=True))
        self.assertEqual(record["kernel"], name)

    def test_table_unknown_attribution(self):
        raw = _raw(16, ts=1, dur=1, tid=1)
        out = self._emit(_events.AttributedEvent(raw, _bpf.API_BY_ID[16], None),
                         json_output=False)
        self.assertIn("<unknown>", out)

    def test_table_row_replaces_control_characters(self):
        # Kernel, span and file names come from the target. A newline or an
        # escape sequence in one must not forge rows or drive the terminal.
        raw = _raw(1, args=(0xF00, 1, 1, 1, 1, 1, 1, 0, 0),
                   ts=3_600_000_000_000, dur=20_000, tid=1)
        ev = _events.AttributedEvent(
            raw, _bpf.API_BY_ID[1],
            _events.FrameInfo("step", "/p/a\nb.py", 3),
            kernel_at_enqueue="evil\x1b[2J\nFAKE ROW", span="be\x07ll")
        out = self._emit(ev, json_output=False)
        self.assertEqual(out.count("\n"), 2)   # the header and one row
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\x07", out)
        self.assertIn("a?b.py:3", out)
        self.assertIn("kernel='evil?[2J?FAKE ROW'", out)
        self.assertIn("span='be?ll'", out)

    def test_table_row_shows_span_when_present(self):
        raw = _raw(1, args=(0xF00, 256, 1, 1, 128, 1, 1, 0, 0x77),
                   ts=3_600_000_000_000, dur=20_000, tid=1)
        ev = _events.AttributedEvent(raw, _bpf.API_BY_ID[1],
                                     _events.FrameInfo("train_step", "/p/train.py", 31),
                                     kernel_at_enqueue="vec_add", span="forward")
        out = self._emit(ev, json_output=False)
        self.assertIn("span=forward", out)

    def test_table_row_omits_span_when_absent(self):
        raw = _raw(16, ts=1, dur=1, tid=1)
        out = self._emit(_events.AttributedEvent(raw, _bpf.API_BY_ID[16], None),
                         json_output=False)
        self.assertNotIn("span=", out)

    def test_json_schema(self):
        raw = _raw(1, args=(0xF00, 256, 1, 1, 128, 1, 1, 0, 0x77),
                   ts=1_000_000_000, dur=20_000, tid=5)
        ev = _events.AttributedEvent(raw, _bpf.API_BY_ID[1],
                                     _events.FrameInfo("f", "/p/a.py", 2),
                                     kernel_at_enqueue="vec_add", span="forward")
        line = self._emit(ev, json_output=True).strip()
        rec = json.loads(line)
        self.assertEqual(
            sorted(rec),
            ["api", "details", "duration_ns", "file", "function", "kernel",
             "line", "pid", "return_code", "span", "tid", "timestamp"])
        self.assertEqual(rec["span"], "forward")
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
        self.assertIsNone(rec["span"])

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


class PrivateDashboardTest(unittest.TestCase):
    """`--web` starts the in-memory dashboard itself; the token stays internal."""

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    @contextlib.contextmanager
    def _quiet_stderr_fd():
        # The runner inherits fd 2 and prints its URL there; keep test output clean.
        saved = os.dup(2)
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)

    def _start(self, port=0):
        with self._quiet_stderr_fd():
            dashboard = metagross._start_private_dashboard(None, port)
        self.addCleanup(metagross._stop_dashboard, dashboard)
        return dashboard

    @staticmethod
    def _alive(pid):
        try:
            with open(f"/proc/{pid}/status") as status:
                return "State:\tZ" not in status.read()
        except FileNotFoundError:
            return False

    def test_receives_a_capture_with_the_internal_token(self):
        dashboard = self._start()
        publisher = _publish.DashboardPublisher(
            dashboard.port, dashboard.token, "workload.py"
        )
        self.addCleanup(publisher.close)
        publisher.start()
        publisher.offer(DashboardPublisherTest._record())
        result = publisher.finish(DashboardPublisherTest._summary(1))
        self.assertIsNone(result.error)
        self.assertEqual(result.dropped_events, 0)

    def test_rejects_any_other_token(self):
        dashboard = self._start()
        publisher = _publish.DashboardPublisher(
            dashboard.port, "other-token-" + ("y" * 32), "workload.py"
        )
        self.addCleanup(publisher.close)
        with self.assertRaisesRegex(_publish.DashboardPublishError, "HTTP 401"):
            publisher.start()

    def test_token_stays_out_of_environment_and_argv(self):
        inherited = "inherited-" + ("t" * 32)
        with mock.patch.dict(os.environ, {"METAGROSS_DASHBOARD_TOKEN": inherited}):
            dashboard = self._start()
        pid = dashboard.process.pid
        with open(f"/proc/{pid}/environ", "rb") as environ_file:
            environ = environ_file.read()
        with open(f"/proc/{pid}/cmdline", "rb") as cmdline_file:
            cmdline = cmdline_file.read()
        for secret in (dashboard.token, inherited):
            self.assertNotIn(secret.encode(), environ)
            self.assertNotIn(secret.encode(), cmdline)

    def test_runs_in_its_own_session_away_from_terminal_ctrl_c(self):
        dashboard = self._start()
        self.assertEqual(os.getsid(dashboard.process.pid), dashboard.process.pid)

    def test_dies_with_the_controller(self):
        helper = subprocess.Popen(
            [sys.executable, "-B", "-c",
             "import metagross, time; "
             "d = metagross._start_private_dashboard(None, 0); "
             "print(d.process.pid, flush=True); time.sleep(30)"],
            cwd=self.REPO, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self.addCleanup(helper.stdout.close)
        pid = int(helper.stdout.readline())
        helper.kill()
        helper.wait()
        deadline = time.monotonic() + 5
        while self._alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(self._alive(pid))

    def test_busy_port_fails_and_reaps_the_runner(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            with mock.patch("metagross._stop_process",
                            wraps=metagross._stop_process) as stop, \
                    self._quiet_stderr_fd(), \
                    self.assertRaisesRegex(MetagrossError, "cannot start the web dashboard"):
                metagross._start_private_dashboard(None, busy.getsockname()[1])
        self.assertIsNotNone(stop.call_args.args[0].returncode)

    def test_serve_until_stopped_ends_on_interrupt_and_restores_sigterm(self):
        import signal as signal_module

        before = signal_module.getsignal(signal_module.SIGTERM)
        process = mock.Mock(wait=mock.Mock(side_effect=KeyboardInterrupt))
        with contextlib.redirect_stderr(io.StringIO()) as err:
            metagross._serve_until_stopped(metagross._Dashboard(process, 1, "t"))
        self.assertIn("Press Ctrl-C", err.getvalue())
        self.assertIs(signal_module.getsignal(signal_module.SIGTERM), before)

    def _run_live_web(self, trace_effect, cfg=None):
        cfg = cfg or Config(script=__file__, project_root=os.path.dirname(__file__),
                            web=True, allow_root_target=True)
        fake = metagross._Dashboard(mock.Mock(), 1, "t" * 43)
        with mock.patch("metagross.os.geteuid", return_value=0), \
                mock.patch("metagross.validate_sudo", return_value=None), \
                mock.patch("metagross._start_private_dashboard",
                           return_value=fake) as start, \
                mock.patch("metagross._trace", side_effect=trace_effect) as trace, \
                mock.patch("metagross._serve_until_stopped") as serve, \
                mock.patch("metagross._stop_dashboard") as stop, \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                result = metagross.run_live(cfg)
            except MetagrossError as exc:
                result = exc
        return result, fake, start, trace, serve, stop

    def test_run_live_refuses_a_root_target_unless_allowed(self):
        # No sudo caller to drop to: the script would run as root.
        cfg = Config(script=__file__, project_root=os.path.dirname(__file__),
                     web=True)
        result, _, start, trace, _, _ = self._run_live_web([0], cfg)
        self.assertIsInstance(result, MetagrossError)
        self.assertIn("--allow-root-target", str(result))
        start.assert_not_called()
        trace.assert_not_called()

    def test_run_live_serves_then_returns_the_target_status(self):
        result, fake, start, trace, serve, stop = self._run_live_web([3])
        self.assertEqual(result, 3)
        start.assert_called_once_with(None, 8765)
        self.assertEqual(trace.call_args.args[4:], (1, "t" * 43))
        self.assertEqual(trace.call_args.kwargs, {"stop_requests": []})
        serve.assert_called_once_with(fake)
        stop.assert_called_once_with(fake)

    def test_run_live_does_not_keep_serving_after_sigterm(self):
        # `docker stop` or `timeout` asked for everything to end.
        def trace(*_args, stop_requests):
            stop_requests.append(signal.SIGTERM)
            return 143

        result, fake, _, _, serve, stop = self._run_live_web(trace)
        self.assertEqual(result, 143)
        serve.assert_not_called()
        stop.assert_called_once_with(fake)

    def test_run_live_stops_the_dashboard_when_tracing_fails(self):
        result, fake, start, _, serve, stop = self._run_live_web(
            MetagrossError("attach failed"),
            Config(script=__file__, project_root=os.path.dirname(__file__),
                   web=True, web_port=0, allow_root_target=True),
        )
        self.assertIsInstance(result, MetagrossError)
        start.assert_called_once_with(None, 0)
        serve.assert_not_called()
        stop.assert_called_once_with(fake)


class PidNamespaceTest(unittest.TestCase):
    def test_only_the_initial_pid_namespace_is_accepted(self):
        with mock.patch("metagross.os.stat",
                        return_value=mock.Mock(st_ino=0xEFFFFFFC)):
            self.assertTrue(_real_in_host_pid_namespace())
        with mock.patch("metagross.os.stat",
                        return_value=mock.Mock(st_ino=4026535076)):
            self.assertFalse(_real_in_host_pid_namespace())
        with mock.patch("metagross.os.stat", side_effect=OSError):
            self.assertTrue(_real_in_host_pid_namespace())

    def test_tracing_is_refused_outside_the_host_pid_namespace(self):
        # There the probes would match no process: an empty capture that
        # looks complete.
        cfg = Config(script=__file__, project_root=os.path.dirname(__file__))
        with mock.patch("metagross.os.geteuid", return_value=0), \
                mock.patch("metagross._in_host_pid_namespace",
                           return_value=False), \
                mock.patch("metagross.os.fork",
                           side_effect=AssertionError("must not fork")):
            with self.assertRaisesRegex(MetagrossError, "--pid=host"):
                metagross.run_live(cfg)


class ArchitectureTest(unittest.TestCase):
    def test_tracing_is_refused_on_other_architectures(self):
        # The launch probes read the x86-64 stack layout; on arm64 they
        # would report wrong block, shared-memory and stream values.
        cfg = Config(script=__file__, project_root=os.path.dirname(__file__))
        with mock.patch("metagross.os.geteuid", return_value=0), \
                mock.patch("metagross._machine", return_value="aarch64"), \
                mock.patch("metagross.os.fork",
                           side_effect=AssertionError("must not fork")):
            with self.assertRaisesRegex(MetagrossError,
                                        "only x86-64.*aarch64"):
                metagross.run_live(cfg)

    def test_unprivileged_commands_work_on_any_architecture(self):
        with mock.patch("metagross._machine", return_value="aarch64"), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(metagross.main(["--ebpf"]), 0)
            self.assertEqual(metagross.main(["--version"]), 0)
        self.assertIn("enter_cuLaunchKernel", out.getvalue())


class RunLiveInitFailureTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _run(self, bpf_factory, output_path=None):
        cfg = Config(script=__file__, project_root=os.path.dirname(__file__),
                     output_path=output_path)
        creds = Credentials(os.getuid(), os.getgid(), "fixture", self.dir.name)
        killed, reaped, writes = [], [], []

        def fake_write(fd, data):
            writes.append(fd)
            return len(data)

        with mock.patch("metagross.os.geteuid", return_value=0), \
                mock.patch("metagross.validate_sudo", return_value=creds), \
                mock.patch("metagross._bpf.find_libcuda", return_value="/unused"), \
                mock.patch.dict("sys.modules", {"bcc": mock.Mock(BPF=bpf_factory)}), \
                mock.patch("metagross._bpf.dlsym_resolver", return_value=lambda s: 1), \
                mock.patch("metagross._bpf.resolve_attachments",
                           return_value=[_stub_attachment()]), \
                mock.patch("metagross._child_main",
                           side_effect=AssertionError("child path must not run")), \
                mock.patch("metagross.os.fork", return_value=4242), \
                mock.patch("metagross.os.kill", side_effect=lambda p, s: killed.append(p)), \
                mock.patch("metagross.os.waitpid",
                           side_effect=lambda p, f=0: reaped.append(p) or (p, 0)), \
                mock.patch("metagross.os.write", side_effect=fake_write):
            with self.assertRaisesRegex(MetagrossError, "ring buffer"):
                metagross.run_live(cfg)
        return killed, reaped, writes

    def test_ring_open_failure_kills_child_before_release(self):
        class FakeBPF:
            def __init__(self, *a, **k): self._maps = {"events": self}
            def __getitem__(self, key): return self
            def attach_uprobe(self, **k): pass
            def attach_uretprobe(self, **k): pass
            def open_ring_buffer(self, cb): raise RuntimeError("ring buffer open failed")
            def cleanup(self): FakeBPF.cleaned = True
        FakeBPF.cleaned = False
        killed, reaped, writes = self._run(FakeBPF)
        self.assertIn(4242, killed)          # child was signalled
        self.assertIn(4242, reaped)          # and reaped
        self.assertTrue(FakeBPF.cleaned)     # BPF object torn down
        self.assertEqual(writes, [])         # barrier was never written to

    def test_a_failed_start_keeps_the_previous_capture(self):
        class FakeBPF:
            def __init__(self, *a, **k): pass
            def __getitem__(self, key): return self
            def attach_uprobe(self, **k): pass
            def attach_uretprobe(self, **k): pass
            def open_ring_buffer(self, cb): raise RuntimeError("ring buffer open failed")
            def cleanup(self): pass
        os.chmod(self.dir.name, 0o700)
        trace = os.path.join(self.dir.name, "trace.jsonl")
        with open(trace, "wb") as handle:
            handle.write(b"the previous capture")
        self._run(FakeBPF, output_path=trace)
        with open(trace, "rb") as handle:
            self.assertEqual(handle.read(), b"the previous capture")


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
