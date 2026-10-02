# metagross/_target.py
"""Private target runner with normal Python interpreter shutdown."""

import os
import sys
import types

from metagross import _profile


def main() -> None:
    """Install attribution and execute the script in the fresh interpreter."""
    profile_fd, drops_fd, project_root, attribution, script = sys.argv[1:6]
    profile_fd, drops_fd = int(profile_fd), int(drops_fd)
    os.set_inheritable(profile_fd, False)
    os.set_inheritable(drops_fd, False)
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
