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
/usr/bin/python3 -B -m unittest -v \
  test_target \
  test_metagross.ParseArgsTest \
  test_metagross.RendererTest \
  test_metagross.DashboardPublisherTest \
  test_viewer.WebDashboardTest \
  test_viewer.ViewerRoutingTest
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
synchronization, process credentials, output file handling, direct dashboard
delivery and startup rejection, and eBPF cleanup. When `bpftool` is available,
it should also verify no extra BPF program remains after shutdown.

## Direct Docker dashboard gate

Rebuild the source-baked image. In the host terminal, generate one token and
start the loopback-only receiver:

```sh
export METAGROSS_DASHBOARD_TOKEN="$(/usr/bin/python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
printf 'Copy this token into the Docker terminal: %s\n' "$METAGROSS_DASHBOARD_TOKEN"
/usr/bin/python3 -B -m metagross view --web --receive --port 8765
```

In the Docker terminal, export the copied token, rebuild, and run:

```sh
docker build \
  --build-arg TARGET_UID="$(id -u)" \
  --build-arg TARGET_GID="$(id -g)" \
  -f examples/docker/Dockerfile \
  -t metagross-pytorch .

docker run --rm \
  --gpus all \
  --privileged \
  --pid=host \
  --network host \
  -e METAGROSS_DASHBOARD_TOKEN \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v /sys/kernel/tracing:/sys/kernel/tracing \
  metagross-pytorch \
  --dashboard-port 8765 \
  --project-root /workspace/workloads \
  /workspace/workloads/basic_tensor_ops.py
```

Do not mount `/traces` or pass output options for this gate. In a real browser,
verify `WAITING → LIVE → COMPLETE`, timeline growth without reload, launch/copy/
memory/sync events, attributed workload functions, target checksum stdout, and
the container's target exit status. `--network host` removes Docker network
isolation; `--pid=host` remains separately necessary for eBPF process identity.
The receiver is ephemeral: stopping it loses state, and a new authorized run
replaces the retained capture. Explicit `--output` and `--summary-output` remain
the durable path.

Open the host receiver's complete private URL with its viewer-token fragment.
Verify the fragment disappears, same-tab refresh retains access, and restarting
the receiver requires its new private URL. The viewer token must differ from
the producer secret copied into Docker. Missing, wrong, or duplicate bearer
credentials must receive 401 from `/api/state`, including HEAD requests, while
static assets remain readable without trace data or credentials. A viewer token
must not authorize capture POSTs, and the producer token must not read state.
`test_viewer.WebDashboardTest` exercises the real HTTP boundary and process
restart; when Node.js is installed it also executes the shipped browser script
to check fragment cleanup, session storage, and exactly one bearer header.
Node.js is optional for tests and is not a dashboard runtime dependency.

Also run two failures. A wrong token must return tracer failure before any
target sentinel. Stopping the receiver during a longer workload must produce
one prefixed controller warning without hanging or changing target stdout,
stderr, or exit status. Restart the receiver with the correct token and run
again to confirm replacement rather than count merging.

## What to test by change type

- CLI parsing or validation: parse/usage tests plus README examples if affected.
- Output path handling: ownership, symlink, regular-file, create/truncate, and
  permission tests.
- Child process behavior: run `test_target` for the unprivileged exec boundary,
  startup barrier, PID preservation, normal shutdown, `atexit`, non-daemon
  threads, buffered stdout/stderr, exceptions, exit codes, signals, argument
  pass-through, environment, and descriptor inheritance. The live gate remains
  required to verify probe attachment across exec and actual credential drops.
- eBPF source changes: generated source tests and, when possible, live gate.
- CUDA API additions: unit tests for raw event enrichment, rendering, reference
  tables, and live gate when available.
- Attribution changes: profile codec, timeline, joiner ordering, unknown-frame,
  and cross-thread tests.
- Rendering/schema changes: table snapshots/assertions, JSONL schema assertions,
  and README updates.
- Viewer changes: unprivileged parser/model tests, bounded-state assertions,
  control-character sanitization, deterministic width/height snapshots, CLI
  routing before root/BCC validation, partial-line and rotation tests for live
  modes, a pseudo-terminal smoke test for curses behavior, loopback HTTP/API
  tests, and a real-browser layout check when practical.

## Handoff expectations

In the final response, say exactly what was run. If the live integration gate was
not run, say why, for example:

> Not run: live integration gate requires sudo, BCC, CUDA, and an NVIDIA driver
> on the host.

Do not imply live tracing was verified unless the live gate actually ran.
