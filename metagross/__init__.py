# metagross/__init__.py
"""Metagross: eBPF GPU-call tracing for one Python script."""
from __future__ import annotations

import ctypes as ct
import dataclasses
import fcntl
import io
import os
import pwd
import runpy
import selectors
import signal
import stat
import sys
import time
import traceback
from typing import NoReturn


class MetagrossError(Exception):
    """Base class for reported failures (exit 1)."""


class UsageError(MetagrossError):
    """Invalid command-line syntax (exit 2)."""


@dataclasses.dataclass
class Config:
    json_output: bool = False
    output_path: str | None = None
    project_root: str = "."
    dump_ebpf: bool = False
    script: str | None = None
    script_args: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class Credentials:
    uid: int
    gid: int
    user: str
    home: str


_USAGE = (
    "usage: sudo /usr/bin/python3 -m metagross [--json] [--output FILE]\n"
    "           [--project-root DIR] script.py [script arguments...]\n"
    "       /usr/bin/python3 -m metagross --ebpf\n"
)


def parse_args(argv: list[str]) -> Config:
    cfg = Config()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith("--"):
            cfg.script = arg
            cfg.script_args = list(argv[i + 1:])
            return cfg
        if arg == "--json":
            cfg.json_output = True
        elif arg == "--ebpf":
            cfg.dump_ebpf = True
        elif arg in ("--output", "--project-root"):
            if i + 1 >= len(argv):
                raise UsageError(f"{arg} requires a value")
            value = argv[i + 1]
            i += 1
            if arg == "--output":
                cfg.output_path = value
            else:
                cfg.project_root = value
        else:
            raise UsageError(f"unknown option: {arg}")
        i += 1
    if cfg.dump_ebpf:
        return cfg
    raise UsageError("missing target script")


def validate_sudo(environ, invoker_lookup) -> Credentials | None:
    keys = ("SUDO_UID", "SUDO_GID", "SUDO_USER")
    present = [k for k in keys if k in environ]
    if not present:
        return None
    if len(present) != len(keys):
        raise MetagrossError("incomplete sudo metadata: "
                             + ", ".join(sorted(set(keys) - set(present))))
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
            f"env {uid}:{gid} vs passwd {pw.pw_uid}:{pw.pw_gid}")
    return Credentials(uid, gid, user, pw.pw_dir)


def exit_status_from_wait(status: int) -> int:
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return os.WEXITSTATUS(status)


def open_trace_output(path: str, uid: int, gid: int):
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.chown(path, uid, gid)
        return os.fdopen(fd, "wb")
    except FileExistsError:
        pass
    except OSError as exc:
        raise MetagrossError(f"cannot create output {path!r}: {exc}") from None
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise MetagrossError(f"output {path!r} is not a regular file")
    if info.st_uid != uid:
        raise MetagrossError(f"output {path!r} is not owned by uid {uid}")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW)
    except OSError as exc:
        raise MetagrossError(f"cannot open output {path!r}: {exc}") from None
    return os.fdopen(fd, "wb")


def _validate_script(script: str, project_root: str) -> str:
    real = os.path.realpath(script)
    root = os.path.realpath(project_root)
    if not real.endswith(".py"):
        raise MetagrossError(f"target must be a .py file: {script!r}")
    if not os.path.isfile(real):
        raise MetagrossError(f"target script not found: {script!r}")
    if not real.startswith(root + os.sep):
        raise MetagrossError(
            f"target {script!r} resolves outside project root {root!r}")
    return real


def _exit_flushed(code: int) -> NoReturn:
    # os._exit skips stdio flushing; the target's own stdout/stderr (and
    # anything we just wrote to them) must reach the parent's pipes.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except OSError:
            pass
    os._exit(code)


def _child_main(script, script_args, creds, barrier_r, profile_w,
                project_root) -> NoReturn:
    """Run post-fork in the traced child. Never returns to the caller."""
    from metagross import _profile

    if creds is not None:
        os.setgroups(os.getgrouplist(creds.user, creds.gid))
        os.setgid(creds.gid)
        os.setuid(creds.uid)
        os.environ["HOME"] = creds.home
        os.environ["USER"] = creds.user
        os.environ["LOGNAME"] = creds.user

    # Close-on-exec: still usable by this process (fork already handed us
    # the fd), but a subprocess the target execs will not inherit it.
    flags = fcntl.fcntl(profile_w, fcntl.F_GETFD)
    fcntl.fcntl(profile_w, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)

    if os.read(barrier_r, 1) == b"":
        _exit_flushed(1)  # parent died before releasing the barrier
    os.close(barrier_r)

    _profile.install(profile_w, project_root)

    sys.argv = [script, *script_args]
    sys.path[0] = os.path.dirname(script)
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            code = 0
        elif not isinstance(code, int):
            sys.stderr.write(f"{code}\n")
            code = 1
        _exit_flushed(code)
    except KeyboardInterrupt:
        _exit_flushed(130)
    except BaseException:
        traceback.print_exc()
        _exit_flushed(1)
    _exit_flushed(0)


def run_live(cfg: Config) -> int:
    # 1. Root check; validate sudo metadata, or warn and run as root.
    if os.geteuid() != 0:
        raise MetagrossError("must run as root (use sudo)")
    creds = validate_sudo(os.environ, pwd.getpwnam)
    if creds is None:
        print("metagross: running target as root", file=sys.stderr)
        uid, gid = 0, 0
    else:
        uid, gid = creds.uid, creds.gid

    # 2. Validate the target, resolve libcuda, and load bcc (lazily, since
    # it is a privileged/optional dependency unneeded by unprivileged paths).
    from metagross import _bpf
    script = _validate_script(cfg.script, cfg.project_root)
    lib_path = _bpf.find_libcuda()
    try:
        from bcc import BPF
    except ImportError as exc:
        raise MetagrossError(
            "bcc (BPF Compiler Collection) not available; "
            f"install python3-bpfcc: {exc}") from None

    # 3. Clock anchor and the trace output stream. Renderer writes str;
    # open_trace_output returns a binary file, so the wrapper is mandatory.
    wall_minus_mono_ns = time.time_ns() - time.monotonic_ns()
    if cfg.output_path:
        stream = io.TextIOWrapper(
            open_trace_output(cfg.output_path, uid, gid), encoding="utf-8")
    else:
        stream = sys.stderr

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
            _child_main(script, cfg.script_args, creds, barrier_r, profile_w,
                        cfg.project_root)
        except BaseException:
            traceback.print_exc()
        finally:
            _exit_flushed(1)
    os.close(profile_w)
    os.close(barrier_r)

    # 6. Attach probes filtered to the child's exact TGID.
    from metagross import _events, _profile
    try:
        b = BPF(text=_bpf.build_source(pid))
        resolver = _bpf.dlsym_resolver(lib_path)
        attachments = _bpf.resolve_attachments(_bpf.APIS, resolver)
        for att in attachments:
            b.attach_uprobe(name=lib_path, sym=att.symbol,
                            fn_name=f"enter_{att.api.base}", pid=pid)
            b.attach_uretprobe(name=lib_path, sym=att.symbol,
                               fn_name=f"exit_{att.api.base}", pid=pid)
    except Exception as exc:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        os.waitpid(pid, 0)
        os.close(barrier_w)
        raise MetagrossError(f"failed to attach probes: {exc}") from exc

    # 7. Release the barrier; the child starts running the target script.
    try:
        os.write(barrier_w, b"\x01")
    except OSError:
        os.waitpid(pid, 0)
        raise MetagrossError("target exited before tracing began") from None
    os.close(barrier_w)

    # 8. Event loop: drain GPU events and profiling records, attribute,
    # render, and watch for the child's exit.
    joiner = _events.Joiner()
    renderer = _events.Renderer(stream, cfg.json_output, wall_minus_mono_ns,
                                pid, joiner)
    renderer.header()
    render_broken = False

    def _emit_all(events):
        nonlocal render_broken
        for ev in events:
            if render_broken:
                continue
            try:
                renderer.emit(ev)
            except MetagrossError as exc:
                print(f"metagross: {exc}", file=sys.stderr)
                render_broken = True

    def _on_ring_event(_ctx, data, size):
        joiner.on_gpu_event(_bpf.decode_event(ct.string_at(data, size)))

    b["events"].open_ring_buffer(_on_ring_event)

    reader = _profile.RecordReader()
    selector = selectors.DefaultSelector()
    selector.register(profile_r, selectors.EVENT_READ)

    def _drain_profile():
        for key, _mask in selector.select(0):
            data = os.read(key.fd, 65536)
            if data:
                for rec in reader.feed(data):
                    joiner.on_profile_record(rec)

    status = None
    interrupted = False
    reaped = False
    while True:
        try:
            if not reaped:
                b.ring_buffer_poll(50)
                _drain_profile()
                _emit_all(joiner.flush(time.monotonic_ns()))
                wpid, status = os.waitpid(pid, os.WNOHANG)
                reaped = wpid == pid
            if reaped:
                # Child has exited (status captured above): drain whatever
                # remains once more and stop, regardless of further signals.
                b.ring_buffer_poll(0)
                _drain_profile()
                _emit_all(joiner.flush(time.monotonic_ns(), force=True))
                break
        except KeyboardInterrupt:
            # 10. Forward SIGINT to the child and keep looping; its exit
            # status will yield 130 naturally once it terminates. If the
            # child was already reaped, the exit status is already known,
            # so stop instead of re-waiting on an already-reaped pid.
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
    if lost:
        print(f"metagross: lost {lost} events (ring buffer full)",
             file=sys.stderr)
    if dropped:
        print(f"metagross: dropped {dropped} nested calls", file=sys.stderr)
    b.cleanup()
    if stream is not sys.stderr:
        stream.close()

    # 10. Forward the target's exit status.
    return exit_status_from_wait(status)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        cfg = parse_args(argv)
    except UsageError as exc:
        print(f"metagross: {exc}\n{_USAGE}", file=sys.stderr, end="")
        return 2
    if cfg.dump_ebpf:
        from metagross import _bpf
        print(_bpf.build_source(0))
        return 0
    # 11. Route to the live launcher; report tracer failures as exit 1.
    try:
        return run_live(cfg)
    except MetagrossError as exc:
        print(f"metagross: {exc}", file=sys.stderr)
        return 1
