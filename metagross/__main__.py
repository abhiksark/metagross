# metagross/__main__.py
"""Entry point for `python3 -m metagross` and `python3 /path/to/metagross`."""
import os
import sys

if not __package__:
    # Launched by path: sys.path[0] is this package directory, never the
    # working directory. Point it at the installation instead.
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

import metagross

sys.exit(metagross.main())
