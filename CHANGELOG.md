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
