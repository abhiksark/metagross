# Changelog

User-visible changes to Metagross. Versions follow `metagross.__version__`.

## 0.1.0 (unreleased)

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
