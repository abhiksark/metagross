# metagross/__init__.py
"""Metagross: eBPF GPU-call tracing for one Python script."""

from __future__ import annotations

import os
import sys


def _launched_as_module() -> bool:
    """Return whether this is `python -m metagross`, however it was spelled.

    A program that merely imports Metagross must keep its own import path,
    so the module being launched has to be this one.
    """
    if sys.argv[0] != "-m":  # set only while Python locates the -m module
        return False
    arguments = sys.orig_argv
    for index, argument in enumerate(arguments):
        if not argument.startswith("-") or argument.startswith("--"):
            continue
        if argument.endswith("mmetagross"):  # -mmetagross, -Bmmetagross
            return True
        if argument.endswith("m") and arguments[index + 1:index + 2] == ["metagross"]:
            return True  # -m metagross, -Bm metagross
    return False


def _pin_import_path() -> None:
    """Keep `python -m metagross` from importing out of a foreign directory.

    `-m` puts the working directory first on `sys.path`, and the controller
    normally runs as root. Drop that entry unless it is this installation,
    before anything else is imported.
    """
    if not _launched_as_module():
        return
    try:
        cwd = os.path.realpath(os.getcwd())
    except OSError:
        return
    package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if (sys.path and os.path.realpath(sys.path[0] or ".") == cwd
            and cwd != package_root):
        del sys.path[0]


_pin_import_path()  # must run before the imports below

import contextlib
import ctypes as ct
import dataclasses
import fcntl
import io
import json
import pwd
import select
import signal
import stat
import subprocess
import time
import traceback
from typing import NoReturn


def __getattr__(name: str):
    """Lazily re-export `metagross.span` (PEP 562).

    A plain top-level `from metagross._span import span` would force
    `metagross._profile` into `sys.modules` on every `import metagross`,
    including the unprivileged `--help` path
    (test_target.TopLevelHelpTest.test_help_does_not_import_tracing_modules
    asserts it stays out). `_span` imports only `_profile` -- no cycle with
    this package -- but the import is deferred to first access of `span`
    regardless, to keep the module import itself free of that side effect.
    """
    if name == "span":
        from metagross._span import span

        return span
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class MetagrossError(Exception):
    """Base class for reported failures (exit 1)."""


class UsageError(MetagrossError):
    """Invalid command-line syntax (exit 2)."""


@dataclasses.dataclass
class Config:
    show_help: bool = False
    json_output: bool = False
    output_path: str | None = None
    project_root: str = "."
    dump_ebpf: bool = False
    show_stats: bool = False
    summary_output_path: str | None = None
    python_attribution: bool = True
    trace_families: frozenset[str] | None = None
    dashboard_port: int | None = None
    web: bool = False
    web_port: int | None = None
    script: str | None = None
    script_args: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class Credentials:
    uid: int
    gid: int
    user: str
    home: str


_FINAL_DRAIN_TIMEOUT_S = 2.0
_DASHBOARD_READY_TIMEOUT_S = 5.0
_DEFAULT_WEB_PORT = 8765
_TRACE_FAMILIES = frozenset(("launch", "memory", "copy", "sync"))

_USAGE = (
    "usage: sudo /usr/bin/python3 -m metagross [--json] [--output FILE]\n"
    "           [--stats] [--summary-output FILE] [--project-root DIR]\n"
    "           [--trace FAMILIES] [--no-attribution]\n"
    "           [--dashboard-port PORT | --web [--web-port PORT]]\n"
    "           script.py [script arguments...]\n"
    "       /usr/bin/python3 -m metagross [--trace FAMILIES] --ebpf\n"
    "       /usr/bin/python3 -m metagross view (--snapshot|--follow|--web) TRACE.jsonl\n"
    "       /usr/bin/python3 -m metagross view --web --receive [--port PORT]\n"
    "       /usr/bin/python3 -m metagross [-h|--help]\n"
)


def _parse_trace_families(value: str) -> frozenset[str] | None:
    names = [name.strip() for name in value.split(",")]
    if not names or any(not name for name in names):
        raise UsageError("--trace requires a comma-separated family list")
    selected = frozenset(names)
    if "all" in selected:
        if len(selected) != 1:
            raise UsageError("--trace all cannot be combined with other families")
        return None
    unknown = selected - _TRACE_FAMILIES
    if unknown:
        supported = ", ".join(sorted(_TRACE_FAMILIES))
        raise UsageError(
            f"unknown trace family {sorted(unknown)[0]!r}; supported: {supported}"
        )
    return selected

def _parse_web_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise UsageError("--web-port must be an integer") from None
    if not 0 <= port <= 65_535:
        raise UsageError("--web-port must be between 0 and 65535")
    return port


def _parse_dashboard_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise UsageError("--dashboard-port must be an integer") from None
    if not 1 <= port <= 65_535:
        raise UsageError("--dashboard-port must be between 1 and 65535")
    return port


def parse_args(argv: list[str]) -> Config:
    cfg = Config()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            cfg.show_help = True
            return cfg
        if not arg.startswith("--"):
            cfg.script = arg
            cfg.script_args = list(argv[i + 1 :])
            _check_web_options(cfg)
            return cfg
        if arg == "--json":
            cfg.json_output = True
        elif arg == "--ebpf":
            if cfg.dashboard_port is not None:
                raise UsageError("--dashboard-port cannot be combined with --ebpf")
            cfg.dump_ebpf = True
        elif arg == "--stats":
            cfg.show_stats = True
        elif arg == "--no-attribution":
            cfg.python_attribution = False
        elif arg == "--web":
            cfg.web = True
        elif arg in (
            "--output",
            "--summary-output",
            "--project-root",
            "--trace",
            "--dashboard-port",
            "--web-port",
        ):
            if i + 1 >= len(argv):
                raise UsageError(f"{arg} requires a value")
            value = argv[i + 1]
            i += 1
            if arg == "--output":
                cfg.output_path = value
            elif arg == "--summary-output":
                cfg.summary_output_path = value
            elif arg == "--project-root":
                cfg.project_root = value
            elif arg == "--trace":
                cfg.trace_families = _parse_trace_families(value)
            elif arg == "--web-port":
                cfg.web_port = _parse_web_port(value)
            else:
                if cfg.dump_ebpf:
                    raise UsageError(
                        "--dashboard-port cannot be combined with --ebpf"
                    )
                cfg.dashboard_port = _parse_dashboard_port(value)
        else:
            raise UsageError(f"unknown option: {arg}")
        i += 1
    _check_web_options(cfg)
    if cfg.dump_ebpf:
        return cfg
    raise UsageError("missing target script")


def _check_web_options(cfg: Config) -> None:
    if cfg.web_port is not None and not cfg.web:
        raise UsageError("--web-port requires --web")
    if cfg.web and cfg.dashboard_port is not None:
        raise UsageError("--web cannot be combined with --dashboard-port")
    if cfg.web and cfg.dump_ebpf:
        raise UsageError("--web cannot be combined with --ebpf")


def validate_sudo(environ, invoker_lookup) -> Credentials | None:
    keys = ("SUDO_UID", "SUDO_GID", "SUDO_USER")
    present = [k for k in keys if k in environ]
    if not present:
        return None
    if len(present) != len(keys):
        raise MetagrossError(
            "incomplete sudo metadata: " + ", ".join(sorted(set(keys) - set(present)))
        )
    try:
        uid, gid = int(environ["SUDO_UID"]), int(environ["SUDO_GID"])
    except ValueError as exc:
        raise MetagrossError(f"non-numeric sudo metadata: {exc}") from None
    user = environ["SUDO_USER"]
    if uid == 0:
        return None
    try:
        pw = invoker_lookup(user)
    except KeyError:
        raise MetagrossError(f"unknown sudo user: {user!r}") from None
    if pw.pw_uid != uid or pw.pw_gid != gid:
        raise MetagrossError(
            f"sudo metadata mismatch for {user!r}: "
            f"env {uid}:{gid} vs passwd {pw.pw_uid}:{pw.pw_gid}"
        )
    return Credentials(uid, gid, user, pw.pw_dir)


def exit_status_from_wait(status: int) -> int:
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return os.WEXITSTATUS(status)


def _open_output_parent(path: str, uid: int) -> int:
    """Pin an absolute output parent without traversing untrusted links or owners."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(os.path.sep, flags)
    try:
        parent = os.path.dirname(path)
        # The leading empty component validates the already-open root.
        for component in parent.rstrip(os.path.sep).split(os.path.sep):
            if component:
                next_fd = os.open(component, flags, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            info = os.fstat(fd)
            if info.st_uid not in (0, uid):
                raise MetagrossError(f"output {path!r} has an untrusted parent owner")
            if (info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                    and not info.st_mode & stat.S_ISVTX):
                raise MetagrossError(
                    f"output {path!r} has a writable non-sticky parent directory"
                )
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_trace_output(path: str, uid: int, gid: int, *, truncate: bool = True):
    """Open and validate the actual inode before any ownership or size change."""
    parent_fd = path_fd = fd = None
    try:
        absolute_path = os.path.join(os.getcwd(), path)
        name = os.path.basename(absolute_path)
        if name in ("", os.curdir, os.pardir):
            raise MetagrossError(f"output {path!r} must name a file")
        parent_fd = _open_output_parent(absolute_path, uid)
        flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        try:
            fd = os.open(name, flags | os.O_CREAT | os.O_EXCL, 0o600,
                         dir_fd=parent_fd)
            created = True
        except FileExistsError:
            path_fd = os.open(name, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC,
                              dir_fd=parent_fd)
            info = os.fstat(path_fd)
            if not stat.S_ISREG(info.st_mode):
                raise MetagrossError(f"output {path!r} is not a regular file")
            if info.st_uid != uid:
                raise MetagrossError(f"output {path!r} is not owned by uid {uid}")
            # Reopen the pinned regular inode, even if its pathname changes.
            fd = os.open(f"/proc/self/fd/{path_fd}", flags & ~os.O_NOFOLLOW)
            created = False
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise MetagrossError(f"output {path!r} is not a regular file")
        if created:
            os.fchown(fd, uid, gid)
        elif info.st_uid != uid:
            raise MetagrossError(f"output {path!r} is not owned by uid {uid}")
        if truncate:
            os.ftruncate(fd, 0)
        stream = os.fdopen(fd, "wb")
        fd = None  # The stream owns the descriptor from this point on.
        return stream
    except OSError as exc:
        raise MetagrossError(f"cannot open output {path!r}: {exc}") from None
    finally:
        if fd is not None:
            os.close(fd)
        if path_fd is not None:
            os.close(path_fd)
        if parent_fd is not None:
            os.close(parent_fd)


def _validate_output_paths(
    output_path: str | None, summary_output_path: str | None
) -> None:
    if output_path is None or summary_output_path is None:
        return
    if os.path.realpath(output_path) == os.path.realpath(summary_output_path):
        raise MetagrossError("trace output and summary output must be different files")


def _open_output_streams(output_path, summary_output_path, uid, gid):
    """Validate both sinks before truncation and transfer their open streams."""
    _validate_output_paths(output_path, summary_output_path)
    try:
        with contextlib.ExitStack() as opened:
            streams = []
            for path in (output_path, summary_output_path):
                if not path:
                    streams.append(None)
                    continue
                binary = opened.enter_context(
                    open_trace_output(path, uid, gid, truncate=False)
                )
                streams.append(opened.enter_context(
                    io.TextIOWrapper(binary, encoding="utf-8")
                ))
            trace, summary = streams
            if trace is not None and summary is not None:
                if os.path.samestat(os.fstat(trace.fileno()),
                                    os.fstat(summary.fileno())):
                    raise MetagrossError(
                        "trace output and summary output must be different files"
                    )
            for stream in streams:
                if stream is not None:
                    os.ftruncate(stream.fileno(), 0)
            opened.pop_all()
            return trace if trace is not None else sys.stderr, summary
    except OSError as exc:
        raise MetagrossError(f"cannot prepare output files: {exc}") from None


def _validate_script(script: str, project_root: str) -> str:
    real = os.path.realpath(script)
    root = os.path.realpath(project_root)
    if not real.endswith(".py"):
        raise MetagrossError(f"target must be a .py file: {script!r}")
    if not os.path.isfile(real):
        raise MetagrossError(f"target script not found: {script!r}")
    # root already ends with os.sep only when it is the filesystem root
    # ("/"); appending another separator there ("//") would fail to
    # prefix-match every real path and reject all of them.
    root_prefix = root if root.endswith(os.sep) else root + os.sep
    if not real.startswith(root_prefix):
        raise MetagrossError(
            f"target {script!r} resolves outside project root {root!r}"
        )
    return real


def _exit_flushed(code: int) -> NoReturn:
    # Startup failures occur before exec and cannot finalize the forked
    # controller interpreter. Preserve their diagnostics before os._exit.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except OSError:
            pass
    os._exit(code)


def _drop_privileges(creds) -> None:
    """Drop the process to creds' group set, gid, and uid, in that order."""
    os.setgroups(os.getgrouplist(creds.user, creds.gid))
    os.setgid(creds.gid)
    os.setuid(creds.uid)
    os.environ["HOME"] = creds.home
    os.environ["USER"] = creds.user
    os.environ["LOGNAME"] = creds.user


_PR_SET_NO_NEW_PRIVS = 38


def _forbid_new_privileges() -> None:
    """Keep the target from gaining privileges through setuid programs."""
    libc = ct.CDLL(None, use_errno=True)
    zero = ct.c_ulong(0)
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, ct.c_ulong(1), zero, zero, zero) != 0:
        raise OSError(ct.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")


def _child_main(
    script, script_args, creds, barrier_r, profile_w, project_root, python_attribution
) -> NoReturn:
    """Drop credentials, await attachment, and exec the target interpreter."""
    os.environ.pop("METAGROSS_DASHBOARD_TOKEN", None)

    if creds is not None:
        _drop_privileges(creds)
    _forbid_new_privileges()

    if os.read(barrier_r, 1) == b"":
        _exit_flushed(1)  # parent died before releasing the barrier
    os.close(barrier_r)

    # Pin the runner to this installation even when the target's working
    # directory has no Metagross package. Credentials are already dropped.
    package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    bootstrap = (
        f"import sys; sys.path.insert(0, {package_root!r}); "
        "from metagross._target import main; del sys.path[0]; main()"
    )
    # Only this exec inherits the profile pipe. The runner restores CLOEXEC
    # before running target code so target descendants cannot retain it.
    os.set_inheritable(profile_w, True)
    interpreter_args = subprocess._args_from_interpreter_flags()
    # The stdlib helper omits -u; preserve effective stream buffering too.
    if getattr(sys.stdout, "write_through", False):
        interpreter_args.append("-u")
    os.execv(sys.executable, [
        sys.executable, *interpreter_args, "-c", bootstrap, str(profile_w), project_root,
        "1" if python_attribution else "0", script, *script_args,
    ])


@dataclasses.dataclass(frozen=True)
class _Dashboard:
    process: subprocess.Popen
    port: int
    token: str


def _start_private_dashboard(creds, port: int) -> _Dashboard:
    """Start the in-memory dashboard as the target user and wait until it listens.

    The producer token is generated here and crosses to the runner only
    through an inherited pipe: never the environment, argv, or the terminal.
    """
    import secrets

    token = secrets.token_urlsafe(32)
    token_r, token_w = os.pipe()
    ready_r, ready_w = os.pipe()
    package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    bootstrap = (
        f"import sys; sys.path.insert(0, {package_root!r}); "
        "from metagross._dashboard import main; del sys.path[0]; "
        "sys.exit(main(sys.argv[1:]))"
    )
    env = {k: v for k, v in os.environ.items() if k != "METAGROSS_DASHBOARD_TOKEN"}
    drop = {}
    if creds is not None:
        env.update(HOME=creds.home, USER=creds.user, LOGNAME=creds.user)
        drop = dict(
            user=creds.uid,
            group=creds.gid,
            extra_groups=os.getgrouplist(creds.user, creds.gid),
        )
    try:
        # A new session keeps terminal Ctrl-C (meant for the target) away from
        # the dashboard; stdout is target-owned, so the runner gets /dev/null.
        process = subprocess.Popen(
            [
                sys.executable, *subprocess._args_from_interpreter_flags(),
                "-c", bootstrap,
                str(token_r), str(ready_w), str(port), str(os.getpid()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            pass_fds=(token_r, ready_w),
            cwd="/",
            env=env,
            start_new_session=True,
            **drop,
        )
    except BaseException:
        for fd in (token_w, ready_r):
            os.close(fd)
        raise
    finally:
        for fd in (token_r, ready_w):
            os.close(fd)
    dashboard = None
    try:
        with os.fdopen(token_w, "wb") as token_file:
            token_file.write(token.encode("ascii"))
        bound_port = _read_ready_port(ready_r, _DASHBOARD_READY_TIMEOUT_S)
        if bound_port is not None:
            dashboard = _Dashboard(process, bound_port, token)
    except OSError:
        pass
    finally:
        os.close(ready_r)
        if dashboard is None:
            _stop_process(process)
    if dashboard is None:
        raise MetagrossError("cannot start the web dashboard")
    return dashboard


def _read_ready_port(fd: int, timeout_s: float) -> int | None:
    """Return the port line the runner writes once it listens, or None."""
    data = b""
    deadline = time.monotonic() + timeout_s
    while not data.endswith(b"\n") and len(data) < 16:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            return None
        chunk = os.read(fd, 16)
        if not chunk:
            return None
        data += chunk
    try:
        return int(data)
    except ValueError:
        return None


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=_DASHBOARD_READY_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _stop_dashboard(dashboard: _Dashboard) -> None:
    _stop_process(dashboard.process)


def _serve_until_stopped(dashboard: _Dashboard) -> None:
    """Keep the finished capture viewable until Ctrl-C or SIGTERM."""
    print(
        "metagross: the target exited; the dashboard is still serving. "
        "Press Ctrl-C to stop it.",
        file=sys.stderr,
    )

    def _interrupt(_signum, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, _interrupt)
    try:
        dashboard.process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_live(cfg: Config) -> int:
    # 1. Root check; validate sudo metadata, or warn and run as root.
    if os.geteuid() != 0:
        raise MetagrossError("must run as root (use sudo)")
    dashboard_token = None
    if cfg.dashboard_port is not None:
        from metagross import _publish

        try:
            dashboard_token = _publish.take_dashboard_token(os.environ)
        except _publish.DashboardPublishError as exc:
            raise MetagrossError(str(exc)) from None
    creds = validate_sudo(os.environ, pwd.getpwnam)
    if creds is None:
        print("metagross: running target as root", file=sys.stderr)
        uid, gid = 0, 0
    else:
        uid, gid = creds.uid, creds.gid
    if not cfg.web:
        return _trace(cfg, creds, uid, gid, cfg.dashboard_port, dashboard_token)

    # Start the dashboard before anything with side effects, so a busy port
    # fails before output files are truncated or the target is forked.
    port = _DEFAULT_WEB_PORT if cfg.web_port is None else cfg.web_port
    dashboard = _start_private_dashboard(creds, port)
    try:
        status = _trace(cfg, creds, uid, gid, dashboard.port, dashboard.token)
        _serve_until_stopped(dashboard)
        return status
    finally:
        _stop_dashboard(dashboard)


def _trace(cfg, creds, uid, gid, dashboard_port, dashboard_token) -> int:
    """Attach probes, run the target, and return its exit status."""
    publisher = None

    # 2. Validate the target, resolve libcuda, and load bcc (lazily, since
    # it is a privileged/optional dependency unneeded by unprivileged paths).
    from metagross import _bpf

    script = _validate_script(cfg.script, cfg.project_root)
    lib_path = _bpf.find_libcuda()
    try:
        from bcc import BPF
    except ImportError as exc:
        raise MetagrossError(
            f"bcc (BPF Compiler Collection) not available; install python3-bpfcc: {exc}"
        ) from None

    # 3. Validate both output inodes before truncating either existing file.
    wall_minus_mono_ns = time.time_ns() - time.monotonic_ns()
    stream, summary_stream = _open_output_streams(
        cfg.output_path, cfg.summary_output_path, uid, gid
    )

    # 4. Pipes: profiling records (child -> parent) and the post-attach
    # barrier (parent -> child).
    profile_r, profile_w = os.pipe()
    barrier_r, barrier_w = os.pipe()
    for fd in (profile_r, profile_w, barrier_r, barrier_w):
        os.set_inheritable(fd, False)
    # Controller ruling: bump the profile pipe's capacity beyond the default
    # 64KiB to mitigate its throughput cap; a failed bump is fine.
    try:
        fcntl.fcntl(profile_r, fcntl.F_SETPIPE_SZ, 1 << 20)
    except OSError:
        pass

    # 5. Fork the target. The child drops privileges and blocks on the
    # barrier until the parent has attached its probes.
    pid = os.fork()
    if pid == 0:
        os.close(profile_r)
        os.close(barrier_w)
        try:
            _child_main(
                script,
                cfg.script_args,
                creds,
                barrier_r,
                profile_w,
                cfg.project_root,
                cfg.python_attribution,
            )
        except BaseException:
            traceback.print_exc()
        finally:
            _exit_flushed(1)
    os.close(profile_w)
    os.close(barrier_r)
    os.set_blocking(profile_r, True)

    b = None

    def _cleanup_before_release() -> None:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            os.waitpid(pid, 0)
        except (ChildProcessError, OSError):
            pass
        for fd in (barrier_w, profile_r):
            try:
                os.close(fd)
            except OSError:
                pass
        if b is not None:
            try:
                b.cleanup()
            except BaseException:
                pass
        if publisher is not None:
            publisher.close()
        if summary_stream is not None:
            try:
                summary_stream.close()
            except BaseException:
                pass
        if stream is not sys.stderr:
            try:
                stream.close()
            except BaseException:
                pass

    # 6. Attach probes filtered to the child's exact TGID.
    from metagross import _events, _profile

    try:
        selected_apis = _bpf.select_apis(cfg.trace_families)
        b = BPF(text=_bpf.build_source(pid, selected_apis))
        resolver = _bpf.dlsym_resolver(lib_path)
        attachments = _bpf.resolve_attachments(selected_apis, resolver)
        if not attachments:
            raise MetagrossError("no symbols found for selected trace families")
        for att in attachments:
            b.attach_uprobe(
                name=lib_path, sym=att.symbol, fn_name=f"enter_{att.api.base}", pid=pid
            )
            b.attach_uretprobe(
                name=lib_path, sym=att.symbol, fn_name=f"exit_{att.api.base}", pid=pid
            )
    except BaseException as exc:
        _cleanup_before_release()
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise MetagrossError(f"failed to attach probes: {exc}") from exc

    if dashboard_port is not None:
        from metagross import _publish

        try:
            publisher = _publish.DashboardPublisher(
                dashboard_port,
                dashboard_token,
                os.path.basename(script),
            )
            publisher.start()
        except BaseException as exc:
            _cleanup_before_release()
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise MetagrossError(f"cannot start dashboard delivery: {exc}") from exc

    # 7. Construct capture machinery: joiner/renderer/stats, the ring-buffer
    # callback, and the profile reader. All of this must be ready before
    # the barrier is released, and any failure here must tear down cleanly
    # with the target still blocked, never stalled unkilled on the barrier.
    try:
        joiner = _events.Joiner()
        renderer = _events.Renderer(stream, cfg.json_output, wall_minus_mono_ns, pid)
        stats = (
            _events.CaptureStats()
            if cfg.show_stats or summary_stream is not None or publisher is not None
            else None
        )
        render_broken = False
        trace_failed = False
        publisher_error_reported = False

        def _report_publisher_error(message: str | None = None) -> None:
            nonlocal publisher_error_reported
            if publisher is None or publisher_error_reported:
                return
            error = message if message is not None else publisher.pop_error()
            if error is not None:
                print(f"metagross: {error}", file=sys.stderr)
                publisher_error_reported = True

        def _emit_all(events):
            nonlocal render_broken
            emitted = False
            for event in events:
                enriched = joiner.enrich(event)
                if stats is not None:
                    stats.observe(enriched)
                if publisher is not None:
                    try:
                        publisher.offer(
                            _events.event_record(enriched, wall_minus_mono_ns, pid)
                        )
                    except Exception as exc:
                        publisher.drop_event(
                            "dashboard event normalization failed: "
                            f"{type(exc).__name__}"
                        )
                    _report_publisher_error()
                if render_broken:
                    continue
                try:
                    renderer.emit(enriched)
                    emitted = True
                except MetagrossError as exc:
                    print(f"metagross: {exc}", file=sys.stderr)
                    render_broken = True
            if emitted and not render_broken:
                try:
                    renderer.flush()
                except MetagrossError as exc:
                    print(f"metagross: {exc}", file=sys.stderr)
                    render_broken = True

        def _on_ring_event(_ctx, data, size):
            joiner.on_gpu_event(_bpf.decode_event(ct.string_at(data, size)))

        b["events"].open_ring_buffer(_on_ring_event)
        # Pre-barrier: dup profile_r into a private fd the reader thread owns.
        # A failed os.dup() still runs _cleanup_before_release() below.
        profile_reader = _profile.ProfileReader(profile_r)
    except BaseException as exc:
        _cleanup_before_release()
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise MetagrossError(f"failed to open ring buffer: {exc}") from exc

    # 8. Release the barrier only after probes, dashboard delivery, the ring
    # buffer, and the profile reader are all ready.
    try:
        os.write(barrier_w, b"\x01")
    except OSError:
        if publisher is not None:
            publisher.abort("target exited before tracing began")
        _cleanup_before_release()
        raise MetagrossError("target exited before tracing began") from None
    os.close(barrier_w)
    # This ordering is required: once the reader thread is blocked in
    # os.read, a pre-barrier _cleanup_before_release() that closes profile_r
    # would be closing an fd under a blocked read, which is undefined.
    # Starting only after the barrier guarantees the cleanup path above can
    # never run against a live reader thread.
    profile_reader.start()

    # 9. Event loop: drain GPU events and profiling records, attribute,
    # render, and watch for the child's exit.
    def _reap_and_capture():
        """Ensure the child is dead and reaped; return its wait status."""
        nonlocal status, reaped
        if reaped:
            return status
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        _, status = os.waitpid(pid, 0)
        reaped = True
        return status

    status = None
    interrupted = False
    reaped = False
    lost = 0
    dropped = 0
    lost_profile = 0
    try:
        renderer.header()
        while True:
            try:
                if not reaped:
                    b.ring_buffer_poll(50)
                    for rec in profile_reader.poll():
                        joiner.on_profile_record(rec)
                    _emit_all(joiner.flush(time.monotonic_ns()))
                    wpid, status = os.waitpid(pid, os.WNOHANG)
                    reaped = wpid == pid
                if reaped:
                    # Child has exited (status captured above): drain
                    # whatever remains once more and stop, regardless of
                    # further signals. Use ring_buffer_consume() when
                    # available so events committed just before the child
                    # exited (adaptive-wakeup) are not missed by a plain
                    # poll(0).
                    try:
                        b.ring_buffer_consume()
                    except AttributeError:
                        b.ring_buffer_poll(0)
                    # Normally the write end is already closed (the child
                    # that held it is gone) and this reaches EOF quickly.
                    # But a grandchild the target forked without exec could
                    # still hold profile_w open, so this must never block
                    # forever: bound the wait with a wall-clock deadline and
                    # stop on EOF or timeout, whichever comes first.
                    deadline = time.monotonic() + _FINAL_DRAIN_TIMEOUT_S
                    for rec in profile_reader.drain_to_eof(deadline):
                        joiner.on_profile_record(rec)
                    _emit_all(joiner.flush(time.monotonic_ns(), force=True))
                    break
            except KeyboardInterrupt:
                # 10. Forward SIGINT to the child and keep looping; its exit
                # status will yield 130 naturally once it terminates. If the
                # child was already reaped, the exit status is already
                # known, so stop instead of re-waiting on an already-reaped
                # pid.
                if reaped:
                    break
                if not interrupted:
                    interrupted = True
                    try:
                        os.kill(pid, signal.SIGINT)
                    except OSError:
                        pass
                continue

        # 9. Report loss counters as warnings and detach probes.
        lost = b["counters"][ct.c_int(0)].value
        dropped = b["counters"][ct.c_int(1)].value
        lost_profile = profile_reader.lost_records()
        if lost:
            print(f"metagross: lost {lost} events (ring buffer full)", file=sys.stderr)
        if dropped:
            print(f"metagross: dropped {dropped} nested calls", file=sys.stderr)
        hook_replacements = profile_reader.hook_replacements()
        if hook_replacements:
            print("metagross: the target replaced the profiling hook "
                  f"{hook_replacements} time(s); calls made on that thread "
                  "afterwards are unattributed", file=sys.stderr)
        if lost_profile > hook_replacements:
            print(f"metagross: lost {lost_profile - hook_replacements} profile "
                  "records (writer overrun)", file=sys.stderr)
    except Exception as exc:
        trace_failed = True
        # Any unexpected failure here must not lose the target's exit
        # status: kill and reap the child so its status is captured, warn
        # once, and fall through to report that status below.
        status = _reap_and_capture()
        print(f"metagross: {exc}", file=sys.stderr)
    finally:
        os.close(profile_r)
        b.cleanup()
        if stream is not sys.stderr:
            try:
                stream.close()
            except OSError as exc:
                render_broken = True
                print(f"metagross: {exc}", file=sys.stderr)

    if status is None:
        status = _reap_and_capture()

    target_exit_status = exit_status_from_wait(status)
    if stats is not None:
        summary = stats.snapshot(
            lost_events=lost,
            dropped_nested_calls=dropped,
            observed_outstanding_bytes=joiner.allocs.total_bytes,
            render_failed=render_broken,
            trace_failed=trace_failed,
            lost_profile_records=lost_profile,
        )
        selected_families = (
            _TRACE_FAMILIES if cfg.trace_families is None else cfg.trace_families
        )
        summary["configuration"] = {
            "trace_families": sorted(selected_families),
            "python_attribution": cfg.python_attribution,
            "attached_symbol_variants": len(attachments),
            "attached_probes": len(attachments) * 2,
        }
        summary["target"] = {
            "pid": pid,
            "script": script,
            "exit_status": target_exit_status,
        }
        if cfg.show_stats:
            capture = summary["capture"]
            print(
                "metagross: stats "
                f"events={capture['events']} "
                f"attributed={capture['attributed']} "
                f"unknown={capture['unknown_attribution']} "
                f"errors={capture['cuda_errors']} "
                f"lost={capture['lost_events']} "
                f"dropped={capture['dropped_nested_calls']} "
                f"lost_profile={capture['lost_profile_records']} "
                f"complete={str(summary['complete']).lower()}",
                file=sys.stderr,
            )
        if summary_stream is not None:
            try:
                json.dump(summary, summary_stream, indent=2, sort_keys=True)
                summary_stream.write("\n")
                summary_stream.flush()
            except OSError as exc:
                print(f"metagross: summary output failed: {exc}", file=sys.stderr)
            finally:
                try:
                    summary_stream.close()
                except OSError as exc:
                    print(f"metagross: {exc}", file=sys.stderr)
        if publisher is not None:
            try:
                result = publisher.finish(summary)
                _report_publisher_error(result.error)
            except Exception as exc:
                publisher.abort("dashboard finalization failed")
                _report_publisher_error(
                    f"dashboard finalization failed: {type(exc).__name__}"
                )
            finally:
                publisher.close()

    # 10. Forward the target's exit status.
    return target_exit_status


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "view":
        from metagross import _viewer

        return _viewer.main(argv[1:])
    try:
        cfg = parse_args(argv)
    except UsageError as exc:
        print(f"metagross: {exc}\n{_USAGE}", file=sys.stderr, end="")
        return 2
    if cfg.show_help:
        print(_USAGE, end="")
        return 0
    if cfg.dump_ebpf:
        from metagross import _bpf

        print(_bpf.build_source(0, _bpf.select_apis(cfg.trace_families)))
        return 0
    # 11. Route to the live launcher; report tracer failures as exit 1.
    try:
        return run_live(cfg)
    except MetagrossError as exc:
        print(f"metagross: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Ctrl-C before run_live's own loop is watching (e.g. during BPF
        # compile/attach, or during header/counter prints outside the
        # loop) must still exit 130 instead of an unhandled traceback.
        return 130
