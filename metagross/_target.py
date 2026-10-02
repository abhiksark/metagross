# metagross/_target.py
"""Private target runner with normal Python interpreter shutdown."""

import fcntl
import os
import resource
import sys
import types

from metagross import _profile


def _move_out_of_reach(fd: int) -> int:
    """Move a tracer descriptor to a high number, closed across exec.

    A script that closes the descriptors it inherited and then opens files
    is handed the lowest free numbers. Up here, the tracer's number is not
    one of them, so profile records cannot end up in the script's files.
    """
    soft_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if soft_limit == resource.RLIM_INFINITY:
        soft_limit = 1 << 16
    try:
        moved = fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC,
                            max(fd, min(soft_limit, 1 << 16) - 16))
    except OSError:
        os.set_inheritable(fd, False)
        return fd
    os.close(fd)
    return moved


def main() -> None:
    """Install attribution and execute the script in the fresh interpreter."""
    profile_fd, drops_fd, project_root, attribution, script = sys.argv[1:6]
    profile_fd = _move_out_of_reach(int(profile_fd))
    drops_fd = _move_out_of_reach(int(drops_fd))
    os.environ.pop("METAGROSS_DASHBOARD_TOKEN", None)
    sys.argv = [script, *sys.argv[6:]]
    sys.path[0] = os.path.dirname(script)
    if attribution == "1":
        _profile.install(profile_fd, project_root, drops_fd)
    with open(script, "rb") as source:
        code = compile(source.read(), script, "exec")
    target = types.ModuleType("__main__")
    target.__file__ = script
    target.__cached__ = None
    # Keep the same globals visible to imports during thread joins and atexit.
    sys.modules["__main__"] = target
    exec(code, target.__dict__)


if __name__ == "__main__":
    main()
