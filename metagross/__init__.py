# metagross/__init__.py
"""Metagross: eBPF GPU-call tracing for one Python script."""
from __future__ import annotations

import dataclasses
import os
import sys


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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        cfg = parse_args(argv)
    except UsageError as exc:
        print(f"metagross: {exc}\n{_USAGE}", file=sys.stderr, end="")
        return 2
    # Later tasks route cfg to the ebpf dump or the live launcher.
    raise NotImplementedError(cfg)
