# Changelog

User-visible changes to Metagross. Versions follow `metagross.__version__`.

## Unreleased

- Python 3.14 is tested. The unit suite now also passes inside a container.
- The source archive now carries the tests' fixtures, the examples, and the
  documents the README links, so its unit tests run from the unpacked archive.
  The wheel carries the logo and font notices next to the license.
- `metagross.__all__` names the Python API: `span`, `main`, and
  `__version__`. A checkout between releases reports a `.dev0` version.
- The reference documents how to run without the Docker image (system Python,
  a virtual environment, or your own image) and how to open the built-in
  dashboard from another machine.
- Metagross refuses to trace on an architecture other than x86-64, where it
  would have reported wrong launch arguments.
- Options that take a value accept `--name=value`. An argument such as `-V`
  is reported as an unknown option instead of being taken as the script.
- Fix: an existing `--output` or `--summary-output` file is emptied only when
  the script is about to start, so a run that fails to attach no longer
  destroys the earlier capture.
- The Docker `metagross` command runs `view`, `--help`, `--version`, and
  `--ebpf` without privileges or GPU access, no longer mistakes an option
  value ending in `.py` for the script, passes extra `docker run` options
  from `METAGROSS_DOCKER_ARGS`, and refuses to run from `/workspace`.
- The snapshot, terminal, and browser viewers say why a capture is
  incomplete, for example `7 profile records lost` or `the script used a
  libcuda that was not traced`. The dashboard state gains an
  `incomplete_reasons` list.
- Breaking, security: `--output` and `--summary-output` are refused in a
  directory that does not belong to the invoking user, unless it is a shared
  sticky directory such as `/tmp`. Before, a user allowed to run only
  Metagross as root could create a file they owned in any root-owned
  directory.
- SIGTERM to Metagross during a run, for example from `docker stop` or
  `timeout`, is passed to the script, as Ctrl-C already was. The capture ends
  with its summary instead of being cut off, and `--web` stops serving.
- Fix: the script no longer keeps running unattended when Metagross is
  killed; the kernel sends it SIGTERM.
- Breaking: table rows show durations in the unit that fits (`3.5us` instead
  of `0.00ms`), and the API column is wide enough for every traced API.
- Fix: a browser that closes its connection no longer makes the dashboard
  print a traceback, and viewer option errors no longer name internal
  functions.

## 0.1.1 (2026-10-02)

- Fix: a call is no longer attributed to a stale function when the tracer is
  behind the script or profile records were dropped. It waits until the
  profile stream has been read past it, and is written as `<unknown>` if that
  does not happen within five seconds.
- New summary field `capture.refused_attributions` (`refused=` in `--stats`):
  calls left `<unknown>` because the tracer lacked their profile history. A
  nonzero count marks the capture incomplete.
- Fix: a script that loaded a `libcuda` the probes are not on now marks the
  capture incomplete (`capture.libcuda_mismatch`). The check compares files,
  not path strings, and runs ten times a second.
- Metagross refuses to start outside the host PID namespace, where it used to
  produce an empty capture reported complete.
- Fix: a call from a function beyond the 65,536-function limit is `<unknown>`
  instead of being given to its caller.
- A hook removed with `threading.setprofile()` marks the capture incomplete.
- Profile records are decoded about twice as fast, and finished threads no
  longer accumulate in the tracer.
- Fix: a mapped file with a name that is not valid UTF-8 no longer makes
  the tracer fail and kill the script, and the wrong-`libcuda` warning no
  longer prints control characters from the library path.
- Fix: a call delivered after a slow read of the profile stream keeps its
  function instead of becoming `<unknown>` in a capture reported complete.
- Fix: the error for an output path under a group-writable directory names
  that directory and the remedy, and the Docker guide no longer recommends a
  workaround that did not work.

## 0.1.0 (2026-10-01)

First public preview.

- Trace selected CUDA driver API calls of one Python script with eBPF and
  attribute each call to the project function that made it.
- Table and JSONL output, a final summary, and `--stats`.
- Terminal and browser dashboards, including `--web` from the tracer.
- `metagross.span()` to label regions, tracked per thread and per `asyncio` task.
- Docker image and a `metagross` wrapper that traces a script in the current
  directory.
- `pip install .` adds a `metagross` command; `--version` prints the version.
- Completeness checks: lost events, dropped nested calls, lost profile
  records, a replaced profiling hook, and a script that is killed or leaves
  through `os._exit` each mark the capture incomplete.
- A warning when the script loads a different `libcuda.so.1` than the one
  the probes are attached to.
- CUDA graph replays (`cuGraphLaunch`) appear as one launch row each, with
  the graph and stream handles and no kernel name.
- Kernel names up to 1,023 bytes are kept whole in JSONL and the summary;
  table rows show the first 124 characters followed by `...`.
