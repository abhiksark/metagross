# metagross/__init__.py
"""Metagross: eBPF GPU-call tracing for one Python script."""
from __future__ import annotations

import dataclasses
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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        cfg = parse_args(argv)
    except UsageError as exc:
        print(f"metagross: {exc}\n{_USAGE}", file=sys.stderr, end="")
        return 2
    # Later tasks route cfg to the ebpf dump or the live launcher.
    raise NotImplementedError(cfg)
