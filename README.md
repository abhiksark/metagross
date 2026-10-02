<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <img src="assets/logo-light.svg" alt="" width="240">
</picture>

[![Tests](https://github.com/abhiksark/metagross/actions/workflows/test.yml/badge.svg)](https://github.com/abhiksark/metagross/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20x86--64-lightgrey)
![eBPF](https://img.shields.io/badge/eBPF-BCC-orange)
![CUDA](https://img.shields.io/badge/NVIDIA-CUDA%20driver%20API-76b900)
![Status](https://img.shields.io/badge/status-experimental-yellow)

# Metagross

Metagross shows which function in your Python project triggered each traced
CUDA kernel launch, memory allocation, copy, and synchronization, without
changing your script. It traces selected CUDA driver API calls with eBPF and reports
their host-side elapsed time, not GPU kernel execution time or utilization.

This is an experimental, source-only public preview for tracing one trusted
Python workload on your own machine.

<img src="assets/dashboard-demo.webp" alt="A terminal runs metagross --web run.py and streams CUDA driver calls, each labeled with its Python function; the web dashboard then fills with those calls grouped by function, shows one cuBLAS launch's details and the summary panels, and the video ends on the metagross run.py command and the line Trace it while it serves live traffic" width="1000">

*The command traces the PyTorch pipeline example; the terminal shows a shortened
stream of its trace rows. The dashboard section is a recorded trace of the same
pipeline (537 CUDA driver calls) replayed at about 1.3× its original pace. Calls
that cannot be safely tied to a project function show as `<unknown>`.*

To try the dashboard without root, BCC, CUDA, or a GPU, open the
[sample capture](examples/captures/README.md).

## When Metagross fits

| Question | Tool |
|----------|------|
| Which project function launched, copied, allocated, or synchronized through CUDA? | Metagross |
| How do GPU execution and CPU/GPU overlap behave? | [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) |
| Why is an individual GPU kernel slow? | [Nsight Compute](https://developer.nvidia.com/nsight-compute) |
| Which operators, autograd work, or tensor allocations dominate? | A framework profiler, such as the PyTorch profiler |

## Functionality overview

| Capability | What it does |
|------------|--------------|
| [Driver call tracing](docs/reference.md#traced-api-table) | Reports 18 CUDA driver APIs through eBPF uprobes: kernel and graph launches, allocations and frees, memory copies, and stream, context, and event synchronization. Three more probes read kernel names. Needs no changes to the traced script; `--trace` selects any of the `launch`, `memory`, `copy`, and `sync` families. |
| [Python attribution](docs/reference.md#how-attribution-works) | Assigns each call to the function under the project root that its thread was in, skipping standard-library, installed-package, and Metagross frames. Calls with no safe project frame stay `<unknown>`; `--no-attribution` turns function attribution off. |
| [Call details](docs/reference.md#traced-api-table) | Records the return code, CPU-side duration, and per-API arguments: launch grid, block, shared memory, and stream; byte counts; pointers; and resolved kernel names. |
| [Named regions](docs/reference.md#op-spans) | Labels the calls made inside `with metagross.span("name"):` by the same thread or `asyncio` task, in table and JSONL output. Spans need function attribution and do nothing when the script runs outside a trace. |
| [Live web dashboard](#read-the-dashboard) | Shows the capture in a browser as it runs, started with `--web`: a function-grouped timeline with zoom, pan, and filters, an event table with a detail inspector, and API, function, kernel, and allocation summaries. |
| [Terminal viewers](docs/reference.md#viewer-option-reference) | Prints a static dashboard with `view --snapshot` or follows a growing JSONL file with `view --follow`. Both run without root, BCC, CUDA, or a GPU. |
| [Durable output](docs/reference.md#output) | Writes table rows to stderr by default, stable JSONL with `--json --output`, a versioned summary with `--summary-output`, and a one-line capture report with `--stats`. |
| [Completeness checks](docs/reference.md#viewer-status-reference) | Tracks lost events, dropped nested calls, and lost profile records, and shows `COMPLETE` only when the final summary reports the capture complete and matches the events received. CUDA error counts are reported but do not affect completeness. |
| [Target preservation](docs/reference.md#privilege-and-trust-boundary) | Runs one script as the image's unprivileged account, with its arguments and stdout unchanged and its exit status passed through; a signal becomes `128 + signal number`. |

## Requirements

Run Metagross through its Docker image. The image bundles Metagross, the BCC
bindings, and PyTorch, so the host needs only:

- x86-64 Linux with kernel 5.8 or later, BPF enabled, and headers for the
  running kernel (on Ubuntu, `sudo apt install "linux-headers-$(uname -r)"`).
- An NVIDIA GPU with a driver that supports CUDA 12.4. RTX 50-series and other
  Blackwell GPUs need the CUDA 12.8 build described in the
  [Docker guide](examples/docker/README.md#build).
- Docker with the NVIDIA Container Toolkit, and permission to run privileged
  containers, which is equivalent to root on the host.

<a id="quick-start"></a>
## Quick start

### 1. Build the image

```sh
git clone https://github.com/abhiksark/metagross.git
cd metagross
docker build \
  --build-arg TARGET_UID="$(id -u)" \
  --build-arg TARGET_GID="$(id -g)" \
  -f examples/docker/Dockerfile \
  -t metagross-pytorch .
```

The build arguments make the traced script run as your UID and GID, so the
trace files it writes belong to you.

### 2. Install the `metagross` command

From the repository root:

```sh
mkdir -p ~/.local/bin
ln -s "$PWD/examples/docker/metagross" ~/.local/bin/metagross
```

`~/.local/bin` must be on your `PATH`. The command runs the image with the
privileges and kernel mounts that tracing needs; the
[Docker guide](examples/docker/README.md#use-the-metagross-command) shows the
full `docker run` it issues.

### 3. Trace the included workload

```sh
metagross
```

Metagross prints one table row per traced CUDA call to stderr, and the
workload's checksum goes to stdout. The [Docker guide](examples/docker/README.md)
lists five more workloads.

### 4. Trace your own script with the live dashboard

```sh
cd /path/to/your/project
metagross --web run.py
```

Open the URL it prints and keep it private. The dashboard fills in as your
script runs and keeps the capture after the script exits. Press Ctrl-C to stop
it; the command then exits with your script's status.

- The trace table prints to stderr in both cases; drop `--web` if you do not
  want the dashboard.
- The current directory is the project root: only functions in files under it
  are attributed, and the script must be a regular `.py` file inside it.
- The current directory is mounted read-write at the same path and your script
  runs there, so relative paths work as they do with `python run.py`. Other
  host directories are not mounted, so paths outside it do not resolve. That is
  a convenience limit, not isolation: see [Limits](#limits).
- Put Metagross options before the script path and script arguments after it;
  the arguments reach your script unchanged.
- Your script can import only what the image provides: PyTorch and NumPy.
  Add other packages to [`examples/docker/Dockerfile`](examples/docker/Dockerfile)
  and rebuild.
- With `--web`, the container also runs with `--network host` so your browser
  can reach the dashboard on `127.0.0.1:8765` (change the port with
  `--web-port`). This removes Docker's network isolation for a container that
  already runs privileged in the host PID namespace.
- Set `METAGROSS_IMAGE` to use another image tag, such as a CUDA 12.8 build.
- The capture is kept in memory only. To also save it to files, see
  [Write JSONL to the host](examples/docker/README.md#write-jsonl-to-the-host).

## Read the dashboard

| Area | What it shows |
|------|---------------|
| Metric strip | Events and event rate, attributed percentage, CUDA errors, CPU API time, synchronization time, copied bytes, and current and peak observed allocations. |
| Integrity counters | Lost, dropped, delivery-dropped, and malformed event counts. |
| CUDA API timeline | One lane per function name, plus `<unknown>`, with zoom, pan, and API-family and text filters. |
| Selection details | The selected call's timing, thread, function, file, kernel, and captured arguments. |
| Events view | The most recent calls matching the timeline filters (500 by default), newest first. |
| Summary analysis | Top CUDA APIs, project functions, top kernels, and the observed allocation curve. |

The status in the top right starts at `WAITING` and becomes `LIVE` once the
capture starts. `LIVE` does not mean the capture is complete. Only `COMPLETE` means the
final summary reported the capture complete and matched the events received; `INCOMPLETE`,
`MISMATCH`, or a malformed warning means the capture cannot be treated as
complete. Allocation figures cover driver allocations Metagross observed, not
device-wide memory use or tensor memory.

See the [status reference](docs/reference.md#viewer-status-reference),
[HTTP fields and schema](docs/reference.md#local-http-interface), and
[storage and display bounds](docs/reference.md#storage-and-display-bounds).

## Mark named regions

Optionally label regions of your own code:

```python
import metagross

with metagross.span("step"):
    ...
```

Calls made inside the block by the same thread or `asyncio` task carry `"step"`
in the `span` field of table and JSONL output; the web dashboard does not show
spans. Spans need function attribution and do nothing when the script runs
outside a trace.
See [op spans](docs/reference.md#op-spans) and the annotated
[`examples/quicklook.py`](examples/quicklook.py).

## Other ways to start

- [Five more PyTorch workloads, correctness tests, and an overhead benchmark](examples/docker/README.md).
- [Terminal snapshot and follow viewers](docs/reference.md#viewer-option-reference)
  for saved JSONL captures.

From the repository root, usage help runs on the host without Docker, root,
BCC, CUDA, or a GPU:

```sh
python3 -m metagross --help
python3 -m metagross view --help
```

## Limits

- Metagross traces selected CUDA driver APIs in one launched Python process and
  its Python threads, on x86-64 only. It does not attach to an existing
  process, follow subprocesses, run `python -m` targets, or record
  framework-level events.
- A CUDA graph replay (`cuGraphLaunch`) is one row; the kernels inside the
  graph are not listed. Several other driver APIs are not traced, and events
  do not say which GPU was used.
- Timing is host-side API elapsed time, not GPU execution time or utilization.
- Attribution and kernel names are best effort. Calls from C++ worker threads
  without an active project Python frame show as `<unknown>`.
- Observed driver allocations are not tensor memory. Caching allocators reuse
  them, and VMM and expandable-segment allocations are outside the traced API set.
- Profiling adds overhead. High call rates and nested driver re-entry can lose
  events, so check warnings and the final summary.
- The container runs privileged and shares the host PID namespace, so running
  it is equivalent to root on the host. Metagross loads its probes as root
  inside it, then runs your script as the image's unprivileged account, which
  can still see host processes. Use it only with trusted local code; it is not
  a sandbox or a production monitor. Keep captures and the private viewer URL
  private.

See the [full limits](docs/reference.md#overhead-and-limits) and the
[privilege and trust boundary](docs/reference.md#privilege-and-trust-boundary).

## Documentation

- [Reference](docs/reference.md): all options, schemas, API coverage, file safety,
  viewer states, exit behavior, and extended limits.
- [Examples](examples/README.md): local demonstrations and function descriptions.
- [Example captures](examples/captures/README.md): where the sample capture came
  from and how to open it.
- [Docker guide](examples/docker/README.md): container workflow, workload portfolio,
  and benchmarks.

## Contributing and security

Issues and focused pull requests are welcome. See [Contributing](CONTRIBUTING.md)
for setup, coding conventions, unprivileged tests, and the privileged live gate.
Report vulnerabilities through [GitHub private vulnerability reporting](SECURITY.md),
not public issues. Only the latest preview commit is supported.

## License and naming

Software is available under [Apache-2.0](LICENSE). The three logo SVGs are excluded
from that software license; see [asset terms and font attribution](assets/README.md).
This preview ships from source, with no package release or stable-support promise.

Metagross is also the name of an [official Pokémon character](https://unite.pokemon.com/en-us/pokemon/metagross/).
This independent project is not affiliated with or endorsed by Pokémon, Nintendo,
Creatures, GAME FREAK, or NVIDIA. The software license does not grant rights to
third-party names or marks.
