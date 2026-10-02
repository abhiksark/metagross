# tests/test_target.py
"""Unprivileged execution-boundary tests with real target interpreters."""

import contextlib
import io
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import metagross
from metagross import _profile


class TargetExecutionTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.script = os.path.join(self.directory, "target.py")

    def _start(self, source, args=(), *, attribution=True, setup="", env=None,
               python_args=()):
        with open(self.script, "w", encoding="utf-8") as stream:
            stream.write(source)
        profile = tempfile.TemporaryFile()
        self.addCleanup(profile.close)
        barrier_r, barrier_w = os.pipe()
        drops_fd = os.eventfd(0, os.EFD_NONBLOCK)
        self.addCleanup(os.close, drops_fd)
        code = (
            "import os, sys, atexit, builtins\n"
            "from metagross import _child_main\n"
            "builtins.controller_only = True\n"
            "atexit.register(lambda: print('controller finalizer'))\n"
            "creds = None\n"
            + setup
            + f"_child_main({self.script!r}, {list(args)!r}, creds, "
            f"{barrier_r}, {profile.fileno()}, {drops_fd}, {self.directory!r}, "
            f"{attribution!r})\n"
        )
        try:
            proc = subprocess.Popen(
                [sys.executable, *python_args, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                pass_fds=(barrier_r, profile.fileno(), drops_fd),
                env=env,
            )
        finally:
            os.close(barrier_r)

        def cleanup():
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=5)

        self.addCleanup(cleanup)
        self.addCleanup(os.close, barrier_w)
        return proc, barrier_w, profile

    def _run(self, source, args=(), **kwargs):
        proc, barrier, profile = self._start(source, args, **kwargs)
        os.write(barrier, b"\x01")
        stdout, stderr = proc.communicate(timeout=10)
        profile.seek(0)
        records = _profile.RecordReader().feed(profile.read())
        return proc.returncode, stdout, stderr, records

    def test_normal_shutdown_joins_threads_then_runs_atexit_and_flushes(self):
        source = (
            "import atexit, sys, threading, time\n"
            "def finish():\n"
            "    sys.stdout.write('atexit buffered')\n"
            "    sys.stderr.write('atexit stderr')\n"
            "atexit.register(finish)\n"
            "def worker():\n"
            "    time.sleep(0.1)\n"
            "    print('thread done')\n"
            "threading.Thread(target=worker).start()\n"
            "print('main done')\n"
        )
        for attribution in (True, False):
            with self.subTest(attribution=attribution):
                code, stdout, stderr, records = self._run(
                    source, attribution=attribution
                )
                self.assertEqual(code, 0, stderr)
                self.assertEqual(stdout, "main done\nthread done\natexit buffered")
                self.assertEqual(stderr, "atexit stderr")
                if attribution:
                    funcs = [rec[4] for rec in records if rec[0] == "frame"]
                    self.assertIn("finish", funcs)
                    self.assertIn("worker", funcs)
                else:
                    self.assertEqual(records, [])

    def test_target_is_signalled_when_the_controller_dies(self):
        # PR_GET_PDEATHSIG (2) reads back what the child armed before exec.
        source = (
            "import ctypes\n"
            "armed = ctypes.c_int(0)\n"
            "ctypes.CDLL(None).prctl(2, ctypes.byref(armed), 0, 0, 0)\n"
            "print(armed.value)\n"
        )
        code, stdout, stderr, _ = self._run(source)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(int(stdout), signal.SIGTERM)

    def test_target_main_module_remains_available_during_shutdown(self):
        source = (
            "import atexit, threading, time\n"
            "value = 42\n"
            "def finish():\n"
            "    import __main__\n"
            "    print('atexit', __main__.value)\n"
            "atexit.register(finish)\n"
            "def worker():\n"
            "    global value\n"
            "    time.sleep(0.1)\n"
            "    value += 1\n"
            "    import __main__\n"
            "    print('thread', __main__.value)\n"
            "threading.Thread(target=worker).start()\n"
        )
        for attribution in (True, False):
            with self.subTest(attribution=attribution):
                code, stdout, stderr, _ = self._run(
                    source, attribution=attribution
                )
                bare = subprocess.run(
                    [sys.executable, self.script], capture_output=True,
                    text=True, timeout=10,
                )
                self.assertEqual((bare.returncode, bare.stdout, bare.stderr),
                                 (0, "thread 43\natexit 43\n", ""))
                self.assertEqual((code, stdout, stderr),
                                 (bare.returncode, bare.stdout, bare.stderr))

    def test_exception_and_system_exit_keep_python_shutdown_semantics(self):
        for statement, code, message in (
            ("raise RuntimeError('target failed')", 1, "RuntimeError: target failed"),
            ("raise SystemExit()", 0, ""),
            ("raise SystemExit(42)", 42, ""),
            ("raise SystemExit('exit message')", 1, "exit message"),
        ):
            with self.subTest(statement=statement):
                actual, stdout, stderr, _ = self._run(
                    "import atexit\n"
                    "atexit.register(lambda: print('shutdown'))\n"
                    + statement + "\n"
                )
                self.assertEqual(actual, code, stderr)
                self.assertEqual(stdout, "shutdown\n")
                if message:
                    self.assertIn(message, stderr)
                else:
                    self.assertEqual(stderr, "")

    def test_arguments_environment_and_io_match_bare_python(self):
        source = (
            "import json, os, sys\n"
            "print(json.dumps([sys.argv, os.getcwd(), os.environ['TARGET_VALUE'], "
            "sys.path[0]]))\n"
            "sys.stdout.write('buffered output')\n"
            "sys.stderr.write('target stderr')\n"
        )
        args = ["--help", "-h", "--json", "", "two words", "λ"]
        env = dict(os.environ, TARGET_VALUE="unchanged", PYTHONUNBUFFERED="1")
        code, stdout, stderr, _ = self._run(source, args, env=env)
        bare = subprocess.run(
            [sys.executable, self.script, *args], capture_output=True,
            text=True, env=env, timeout=10,
        )
        self.assertEqual((code, stdout, stderr),
                         (bare.returncode, bare.stdout, bare.stderr))

    def test_interpreter_options_match_bare_python(self):
        source = (
            "import json, sys\n"
            "def documented():\n"
            "    'target docstring'\n"
            "print(json.dumps([sys.flags.optimize, sys.dont_write_bytecode, "
            "sys.stdout.write_through, sys.stderr.write_through, __debug__, "
            "documented.__doc__, sys.warnoptions, sys._xoptions.get('utf8')]))\n"
        )
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("PYTHON")}
        for flags, optimize in (
            ((), 0),
            (("-O", "-B", "-u"), 1),
            (("-OO", "-B", "-u"), 2),
            (("-OBu",), 1),
        ):
            with self.subTest(flags=flags):
                python_args = (*flags, "-W", "error::UserWarning", "-X", "utf8=1")
                code, stdout, stderr, _ = self._run(
                    source, ["-u"], python_args=python_args, env=env,
                )
                bare = subprocess.run(
                    [sys.executable, *python_args, self.script, "-u"],
                    capture_output=True, text=True, env=env, timeout=10,
                )
                self.assertEqual(bare.returncode, 0, bare.stderr)
                self.assertEqual(json.loads(bare.stdout), [
                    optimize, bool(flags), bool(flags), bool(flags),
                    optimize == 0, None if optimize == 2 else "target docstring",
                    ["error::UserWarning"], "1",
                ])
                self.assertEqual((code, stdout, stderr),
                                 (bare.returncode, bare.stdout, bare.stderr))

    def test_exec_keeps_pid_and_discards_controller_state(self):
        proc, barrier, _ = self._start(
            "import builtins, os\n"
            "print(os.getpid(), hasattr(builtins, 'controller_only'))\n"
        )
        os.write(barrier, b"\x01")
        stdout, stderr = proc.communicate(timeout=10)
        self.assertEqual(proc.returncode, 0, stderr)
        self.assertEqual(stdout, f"{proc.pid} False\n")

    def test_target_waits_for_attach_barrier(self):
        proc, barrier, _ = self._start("print('released', flush=True)\n")
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            self.assertEqual(selector.select(0.1), [])
            os.write(barrier, b"\x01")
            self.assertTrue(selector.select(5))
        stdout, stderr = proc.communicate(timeout=10)
        self.assertEqual((proc.returncode, stdout, stderr), (0, "released\n", ""))

    def test_trace_descriptors_do_not_reach_exec_descendants(self):
        # The target holds the profile pipe and the drop counter, nothing else.
        source = (
            "import os, subprocess, sys\n"
            "fds = [int(fd) for fd in os.listdir('/proc/self/fd') "
            "if int(fd) > 2 and os.path.exists('/proc/self/fd/' + fd)]\n"
            "assert len(fds) == 2, fds\n"
            "for fd in fds:\n"
            "    assert not os.get_inheritable(fd)\n"
            "    check = 'import os; assert not os.path.exists(' "
            "+ repr('/proc/self/fd/' + str(fd)) + ')'\n"
            "    subprocess.run([sys.executable, '-c', check], "
            "close_fds=False, check=True)\n"
            "print('closed')\n"
        )
        code, stdout, stderr, _ = self._run(source)
        self.assertEqual((code, stdout, stderr), (0, "closed\n", ""))

    def test_target_environment_removes_dashboard_secret(self):
        code, stdout, stderr, _ = self._run(
            "import os\nprint('METAGROSS_DASHBOARD_TOKEN' in os.environ)\n",
            env=dict(os.environ, METAGROSS_DASHBOARD_TOKEN="private-test-token"),
        )
        self.assertEqual((code, stdout, stderr), (0, "False\n", ""))

    def test_target_cannot_gain_privileges(self):
        source = (
            "for line in open('/proc/self/status'):\n"
            "    if line.startswith('NoNewPrivs'):\n"
            "        print(line.split()[1])\n"
        )
        returncode, stdout, stderr, _ = self._run(source)
        self.assertEqual((returncode, stdout), (0, "1\n"), stderr)

    def test_dropped_credentials_environment_survives_exec(self):
        # Exercise the real exec/environment path without privileged syscalls.
        setup = (
            "from metagross import Credentials\n"
            "creds = Credentials(1234, 5678, 'target-user', '/target-home')\n"
            "os.getgrouplist = lambda user, gid: [gid]\n"
            "os.setgroups = lambda groups: os.environ.update("
            "CREDENTIAL_STEPS='groups')\n"
            "os.setgid = lambda gid: os.environ.update("
            "CREDENTIAL_STEPS=os.environ['CREDENTIAL_STEPS'] + ',gid')\n"
            "os.setuid = lambda uid: os.environ.update("
            "CREDENTIAL_STEPS=os.environ['CREDENTIAL_STEPS'] + ',uid')\n"
        )
        code, stdout, stderr, _ = self._run(
            "import json, os\n"
            "print(json.dumps([os.environ[name] for name in "
            "('HOME', 'USER', 'LOGNAME', 'CREDENTIAL_STEPS')]))\n",
            setup=setup,
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout),
                         ["/target-home", "target-user", "target-user",
                          "groups,gid,uid"])

    def test_runner_is_found_after_working_directory_changes(self):
        code, stdout, stderr, _ = self._run(
            "print('target ran')\n",
            setup=f"os.chdir({self.directory!r})\n",
        )
        self.assertEqual((code, stdout, stderr), (0, "target ran\n", ""))

    def test_signal_death_matches_bare_python(self):
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signum=signum):
                code, _, stderr, _ = self._run(
                    f"import os\nos.kill(os.getpid(), {int(signum)})\n"
                )
                self.assertEqual(code, -signum, stderr)


class ModuleLaunchImportPathTest(unittest.TestCase):
    def _launch(self, cwd, arguments=("-m", "metagross", "--help"),
                on_python_path=True):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        if on_python_path:
            env["PYTHONPATH"] = os.getcwd()
        return subprocess.run(
            [sys.executable, *arguments],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=30)

    def _plant(self, directory, names=("dataclasses", "traceback")):
        """Write stand-ins for modules Metagross imports; return their log."""
        log = os.path.join(directory, "imported.log")
        for name in names:
            with open(os.path.join(directory, name + ".py"), "w") as stream:
                stream.write(f"open({log!r}, 'a').write({name!r})\n"
                             "raise ImportError('planted module ran')\n")
        return log

    def test_foreign_working_directory_is_not_on_the_import_path(self):
        # `python -m` puts the working directory first on sys.path, and the
        # controller normally runs as root.
        for spelling in (("-m", "metagross"), ("-mmetagross",),
                         ("-Bm", "metagross")):
            with self.subTest(spelling=spelling), \
                    tempfile.TemporaryDirectory() as foreign:
                log = self._plant(foreign)
                result = self._launch(foreign, (*spelling, "--help"))
                self.assertFalse(os.path.exists(log), result.stderr)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_launch_by_path_never_imports_from_the_working_directory(self):
        # Under -m the interpreter itself imports runpy or types from the
        # working directory before Metagross runs. Launched by path it must
        # not, so plant those too.
        launcher = os.path.join(os.getcwd(), "metagross")
        with tempfile.TemporaryDirectory() as foreign:
            log = self._plant(
                foreign, ("dataclasses", "traceback", "runpy", "types"))
            result = self._launch(foreign, (launcher, "--help"),
                                  on_python_path=False)
            self.assertFalse(os.path.exists(log), result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)

    def test_checkout_as_working_directory_still_launches(self):
        result = self._launch(os.getcwd())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)

    def test_another_program_that_imports_metagross_keeps_its_path(self):
        # A project launched with -m that uses metagross.span() must still
        # import its own modules from the working directory.
        with tempfile.TemporaryDirectory() as project:
            os.mkdir(os.path.join(project, "app"))
            for name, source in (("app/__init__.py", "import metagross\n"),
                                 ("app/__main__.py", "import helper\n"),
                                 ("helper.py", "print('helper imported')\n")):
                with open(os.path.join(project, name), "w") as stream:
                    stream.write(source)
            result = self._launch(project, ("-m", "app"))
        self.assertEqual((result.returncode, result.stdout),
                         (0, "helper imported\n"), result.stderr)


class TopLevelHelpTest(unittest.TestCase):
    def test_help_before_target_is_unprivileged(self):
        for args in (["-h"], ["--help"], ["--json", "--help"]):
            with self.subTest(args=args):
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), \
                     contextlib.redirect_stderr(stderr), \
                     mock.patch("metagross.run_live", side_effect=AssertionError), \
                     mock.patch.dict(sys.modules, {"bcc": None}):
                    code = metagross.main(args)
                self.assertEqual(code, 0)
                self.assertIn("usage:", stdout.getvalue())
                self.assertEqual(stderr.getvalue(), "")

    def test_version_is_unprivileged_and_matches_the_package(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), \
             mock.patch("metagross.run_live", side_effect=AssertionError):
            code = metagross.main(["--version"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(),
                         f"metagross {metagross.__version__}\n")
        self.assertRegex(metagross.__version__, r"^\d+\.\d+\.\d+(\.dev\d+)?$")

    def test_help_after_target_is_passed_through(self):
        cfg = metagross.parse_args(["target.py", "-h", "--help"])
        self.assertEqual(cfg.script_args, ["-h", "--help"])

    def test_help_does_not_import_tracing_modules(self):
        code = (
            "import sys, metagross\n"
            "assert metagross.main(['--help']) == 0\n"
            "assert not any(name in sys.modules for name in "
            "('bcc', 'metagross._bpf', 'metagross._profile'))\n"
        )
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
