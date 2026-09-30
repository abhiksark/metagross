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

<img src="assets/dashboard-demo.webp" alt="Metagross web dashboard filling with CUDA driver calls from a PyTorch pipeline, grouped by Python function, then showing one cuBLAS launch's details and the summary panels" width="1000">

*Recorded trace of a PyTorch pipeline (537 CUDA driver calls) streamed into the
web dashboard at about 1.3× its original pace. Calls that cannot be safely tied
to a project function show as `<unknown>`.*

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
| [Driver call tracing](docs/reference.md#traced-api-table) | Reports 17 CUDA driver APIs through eBPF uprobes: kernel launches, allocations and frees, memory copies, and stream, context, and event synchronization. Three more probes read kernel names. Needs no changes to the traced script; `--trace` selects any of the `launch`, `memory`, `copy`, and `sync` families. |
| [Python attribution](docs/reference.md#how-attribution-works) | Assigns each call to the function under the project root that was active at API entry, skipping standard-library, installed-package, and Metagross frames. Calls with no safe project frame stay `<unknown>`; `--no-attribution` turns function attribution off. |
| [Call details](docs/reference.md#traced-api-table) | Records the return code, CPU-side duration, and per-API arguments: launch grid, block, shared memory, and stream; byte counts; pointers; and resolved kernel names. |
| [Named regions](docs/reference.md#op-spans) | Labels the calls made inside `with metagross.span("name"):` on the same thread, in table and JSONL output. Spans need function attribution and do nothing when the script runs outside a trace. |
| [Live web dashboard](#read-the-dashboard) | Shows the capture in a browser as it runs: a function-grouped timeline with zoom, pan, and filters, an event table with a detail inspector, and API, function, kernel, and allocation summaries. |
| [Terminal viewers](docs/reference.md#viewer-option-reference) | Prints a static dashboard with `view --snapshot` or follows a growing JSONL file with `view --follow`. Both run without root, BCC, CUDA, or a GPU. |
| [Durable output](docs/reference.md#output) | Writes table rows to stderr by default, stable JSONL with `--json --output`, a versioned summary with `--summary-output`, and a one-line capture report with `--stats`. |
| [Completeness checks](docs/reference.md#viewer-status-reference) | Tracks lost events, dropped nested calls, and lost profile records, and shows `COMPLETE` only when the final summary reports the capture complete and matches the events received. CUDA error counts are reported but do not affect completeness. |
| [Target preservation](docs/reference.md#privilege-and-trust-boundary) | Runs one script as the sudo caller (or as root, with a warning, when started directly as root), with its arguments and stdout unchanged and its exit status passed through; a signal becomes `128 + signal number`. |

## Requirements

Live tracing needs:

- x86-64 Linux with kernel 5.8 or later, BPF enabled, and headers for the
  running kernel.
- An NVIDIA GPU and driver.
- Root through `sudo`.
- The system Python 3.10 or later (`/usr/bin/python3`) with the distribution's
  BCC bindings. Your script runs under this interpreter, so it must be able to
  import your script's dependencies.

On Ubuntu:

```sh
git clone https://github.com/abhiksark/metagross.git
cd metagross
sudo apt install python3-bpfcc "linux-headers-$(uname -r)"
```

For other distributions, see the
[BCC installation guide](https://github.com/iovisor/bcc/blob/master/INSTALL.md).
The viewers and both `--help` commands run without root, BCC, CUDA, or a GPU.

<a id="quick-start"></a>
## Quick start

Metagross runs from source, so run every command below from the repository root.

### 1. Trace the included demo

```sh
sudo /usr/bin/python3 -m metagross examples/gpu_demo.py
```

Metagross writes one table row per traced call to stderr; the demo's own output
stays on stdout. Rows have this shape (timings vary):

```text
TIME        FUNCTION          LOCATION            API             RET  DURATION DETAILS
12:10:03.41 compute           gpu_demo.py:86      LaunchKernel    0    0.05ms   kernel=vec_add grid=8,1,1 block=128,1,1 shared=0 stream=0x0
```

### 2. Trace your own script

```sh
PROJECT_ROOT="/absolute/path/to/your/project"
WORKLOAD="$PROJECT_ROOT/path/to/workload.py"
sudo /usr/bin/python3 -B -m metagross --project-root "$PROJECT_ROOT" "$WORKLOAD"
```

- Set `PROJECT_ROOT` to your project's absolute path; only functions under it
  are attributed.
- Set `WORKLOAD` to a regular `.py` file inside that root.
- Place Metagross options before `"$WORKLOAD"` and script arguments after it;
  the arguments reach your script unchanged.

See the [complete CLI and interpreter contract](docs/reference.md#usage) and the
[prepared-container route](examples/docker/README.md).

### 3. Watch it live in the browser

The tracer writes the capture to a file, and the dashboard follows that file.
Use two terminals. `traces/` is ignored by Git.

1. **Terminal one: start the dashboard.**

   ```sh
   mkdir -p traces
   /usr/bin/python3 -m metagross view --web \
     --summary traces/live-summary.json traces/live.jsonl
   ```

   Open the URL it prints and keep it private. The page shows `WAITING` until
   the trace file appears. To view the dashboard from another machine on a
   trusted internal network, see the
   [viewer access options](docs/reference.md#visual-trace-viewer).

2. **Terminal two: run the tracer.**

   ```sh
   PROJECT_ROOT="/absolute/path/to/your/project"
   WORKLOAD="$PROJECT_ROOT/path/to/workload.py"
   sudo /usr/bin/python3 -B -m metagross --json \
     --output traces/live.jsonl --summary-output traces/live-summary.json \
     --project-root "$PROJECT_ROOT" "$WORKLOAD"
   ```

Rerunning the tracer overwrites both files, and the open page starts over with
the new run. Stop the dashboard with Ctrl-C. The files stay in `traces/`, so you
can reopen them later with `view --web` or `view --snapshot`. They contain
source paths and function names; delete them when you no longer need them.

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

Calls made inside the block on the same thread carry `"step"` in the `span`
field of table and JSONL output; the web dashboard does not show spans. Spans
need function attribution and do nothing when the script runs outside a trace.
See [op spans](docs/reference.md#op-spans) and the annotated
[`examples/quicklook.py`](examples/quicklook.py).

## Other ways to start

- [Prepared Docker workload, alternate workloads, and benchmarks](examples/docker/README.md).
- [Terminal snapshot and follow viewers](docs/reference.md#viewer-option-reference)
  for saved JSONL captures.

Usage help runs without root, BCC, CUDA, or a GPU:

```sh
/usr/bin/python3 -m metagross --help
/usr/bin/python3 -m metagross view --help
```

## Limits

- Metagross traces selected CUDA driver APIs in one launched Python process and
  its Python threads. It does not attach to an existing process, follow
  subprocesses, run `python -m` targets, or record framework-level events.
- Timing is host-side API elapsed time, not GPU execution time or utilization.
- Attribution and kernel names are best effort. Calls from C++ worker threads
  without an active project Python frame show as `<unknown>`.
- Observed driver allocations are not tensor memory. Caching allocators reuse
  them, and VMM and expandable-segment allocations are outside the traced API set.
- Profiling adds overhead. High call rates and nested driver re-entry can lose
  events, so check warnings and the final summary.
- Metagross loads its probes as root, then runs your script as your own user
  when started through `sudo`, or as root with a warning otherwise. It is for trusted local workloads and is not a sandbox or a production
  monitor. Keep captures and the private viewer URL private.

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
Creatures, GAME FREAK, or NVIDIA. Naming clearance or a rename is required before
a stable release. The software license does not grant rights to third-party names
or marks.
