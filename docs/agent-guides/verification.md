# Verification guide

Use the smallest meaningful check while developing, then run the full
unprivileged suite before handoff for Python behavior changes.

## Default gate

Run from the repository root:

```sh
/usr/bin/python3 -m unittest -v
```

This must remain unprivileged. It should not require BCC, CUDA, root, an NVIDIA
GPU, or an NVIDIA driver.

## Focused checks

Use focused tests while iterating:

```sh
/usr/bin/python3 -m unittest -v test_metagross.ParseTest
/usr/bin/python3 -m unittest -v test_metagross.RenderTest
/usr/bin/python3 -m unittest -v test_metagross.JoinerTest
/usr/bin/python3 -m unittest -v test_metagross.ProfileTest
/usr/bin/python3 -m unittest -v test_metagross.BpfSourceTest
```

If class names change, inspect `test_metagross.py` and run the closest affected
classes or individual test methods.

## Live integration gate

Run only when the host has BCC, CUDA, an NVIDIA driver, and sudo/root available,
or when a task touches live tracing, probe attachment, CUDA API capture,
credentials, output ownership, or cleanup:

```sh
sudo env RUN_EBPF_INTEGRATION=1 \
  /usr/bin/python3 -m unittest -v test_metagross.LiveTraceTest
```

or:

```sh
./run_integration.sh
```

The live gate should exercise kernel launches, memory operations,
synchronization, process credentials, output file handling, and eBPF cleanup.
When `bpftool` is available, it should also verify no extra BPF program remains
after shutdown.

## What to test by change type

- CLI parsing or validation: parse/usage tests plus README examples if affected.
- Output path handling: ownership, symlink, regular-file, create/truncate, and
  permission tests.
- Child process behavior: exit-code, signal, stdout/stderr, argument pass-through,
  and environment/drop-privilege tests.
- eBPF source changes: generated source tests and, when possible, live gate.
- CUDA API additions: unit tests for raw event enrichment, rendering, README
  tables, and live gate when available.
- Attribution changes: profile codec, timeline, joiner ordering, unknown-frame,
  and cross-thread tests.
- Rendering/schema changes: table snapshots/assertions, JSONL schema assertions,
  and README updates.

## Handoff expectations

In the final response, say exactly what was run. If the live integration gate was
not run, say why, for example:

> Not run: live integration gate requires sudo, BCC, CUDA, and an NVIDIA driver
> on the host.

Do not imply live tracing was verified unless the live gate actually ran.
