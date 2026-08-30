# Metagross: eBPF GPU-call tracing for one Python script

Metagross runs one local Python script and attributes selected NVIDIA CUDA
driver API calls to the project function that caused them. Use it to identify
which functions launch kernels, move data, allocate or release driver memory,
or wait for synchronization. The target keeps its normal standard output and
standard error. Trace records use a readable table by default or JSONL for
machine processing. Calls made through frameworks such as PyTorch are visible
when they reach the traced CUDA driver exports.

Metagross is currently a source-run local diagnostic tool, not a production
monitor or a replacement for Nsight/CUPTI. The tested target is Ubuntu 22.04,
Linux 6.8, x86_64, `/usr/bin/python3`, BCC, and an NVIDIA CUDA driver. Other
platform combinations are not yet part of a compatibility guarantee.

## Setup

Metagross currently runs directly from a repository checkout. Install the Ubuntu
BCC Python package and ensure the NVIDIA driver is installed:

```sh
sudo apt install python3-bpfcc
```

If BCC cannot find the running kernel's headers, also install
`linux-headers-$(uname -r)`. The target and its dependencies run under
`/usr/bin/python3`, so packages imported by the target must be available to that
interpreter.

The NVIDIA driver is required for live GPU tracing. Imports, generated-source
inspection, and the unprivileged unit suite work without it. BCC and root
privileges are required only for live tracing. Upstream provides the
[BCC installation guide](https://github.com/iovisor/bcc/blob/master/INSTALL.md)
and [Python API source](https://github.com/iovisor/bcc/blob/master/src/python/bcc/__init__.py).

### Dockerized PyTorch examples

A containerized CUDA 12.4/PyTorch image and a suite of model workloads are
available in [`examples/docker/`](examples/docker/README.md). The container
still uses the host NVIDIA driver, kernel, and eBPF facilities, so it requires
the NVIDIA Container Toolkit, host kernel headers and tracefs mounts,
`--privileged`, and `--pid=host`. It is a local development example rather than
an isolation boundary for untrusted code.

## Usage

Run the included demonstration from the repository root:

```sh
sudo /usr/bin/python3 -m metagross examples/gpu_demo.py
```

The complete interface is:

```text
sudo /usr/bin/python3 -m metagross \
  [--json] [--output FILE] [--stats] [--summary-output FILE] \
  [--project-root DIR] [--trace FAMILIES] [--no-attribution] \
  script.py [script arguments...]
```

Metagross options must appear before the script. Everything after the script is
passed to it unchanged. `--project-root` defaults to the current directory.
The script must resolve to a regular `.py` file inside that directory.

`--trace` accepts a comma-separated selection of `launch`, `memory`, `copy`, and
`sync`; `all` is the default. Launch capture automatically includes the internal
probes needed to resolve kernel names. Use `--no-attribution` to skip the Python
profiling hook when CUDA API events are needed without project function names.
Its table output uses `<unknown>` and its JSON attribution fields are `null`.
These controls are useful for reducing startup and target overhead:

```sh
sudo /usr/bin/python3 -m metagross \
  --trace launch --no-attribution \
  examples/gpu_demo.py
```

Without `--output`, trace records go to standard error so the target's standard
output remains unchanged. Diagnostics and target standard error can share that
stream. Use a separate file for reliable machine processing:

```sh
sudo /usr/bin/python3 -m metagross \
  --json --output /tmp/metagross.jsonl \
  examples/gpu_demo.py
```

New trace and summary files have mode `0600` and are owned by the invoking user.
Metagross truncates an existing output only when it is a regular, non-symlink
file owned by that user. Trace and summary output must use different paths.
Under sudo, the controller retains the privileges needed for BPF while the
target runs as the validated invoking user. Direct root execution runs the
target as root and emits a warning.

Inspect generated eBPF C code without importing BCC or requiring root:

```sh
/usr/bin/python3 -m metagross --ebpf
/usr/bin/python3 -m metagross --trace launch --ebpf
```

## How attribution works

The child process uses Python profiling hooks to report project function calls
and returns. For each selected CUDA API, an eBPF uprobe stores call arguments
and its paired uretprobe emits a completed event with the return code and
CPU-side duration. Metagross joins GPU and profile events by native thread ID
and the shared monotonic clock, then attributes each API call to the project
frame that was active at API entry.

Calls in the standard library, site packages, virtual-environment packages,
and Metagross itself do not replace the nearest project frame. A memory
allocation performed inside a standard-library function, for example, remains
attributable to the project function that initiated it. Imported project
modules and Python threads are included.

The target remains behind a pipe barrier until every uprobe and uretprobe
pair is attached for its exact process ID. API calls that occur before any
project frame is active are attributed as unknown rather than suppressed.
Events are held for 100 ms to tolerate cross-CPU and cross-stream delivery
ordering. Long API calls appear after they complete.

## Output

The table begins with this shape:

```text
TIME        FUNCTION          LOCATION            API             RET  DURATION DETAILS
12:10:03.41 compute           gpu_demo.py:86      LaunchKernel    0    0.05ms   kernel=vec_add grid=8,1,1 block=128,1,1 shared=0 stream=0x0
```

Detail strings use shell-safe quoting. `<unknown>` means the profile or eBPF
stream did not contain enough matching data for safe attribution.

JSONL records use this exact top-level schema:

```json
{"timestamp":"2026-08-24T12:10:03.410000+05:30","pid":1234,"tid":1234,"function":"compute","file":"/home/user/project/examples/gpu_demo.py","line":86,"api":"cuLaunchKernel","kernel":"vec_add","return_code":0,"duration_ns":50000,"details":{"grid":"8,1,1","block":"128,1,1","shared":0,"stream":"0x0","function_handle":"0xf00"}}
```

`function`, `file`, and `line` are `null` when attribution is unknown. `file` is
the path reported by the Python code object and is normally absolute; table
output displays only its basename. `line` is the function definition line.
`return_code` is the signed raw CUDA driver `CUresult`. Timestamps are local ISO
8601 values derived from the API call's monotonic start time and the parent
startup wall-clock offset.

For a launch with a resolved name, `kernel` contains that name. For non-launch
events it is `null`. An unresolved launch is shown as `kernel@0x...` in table
output, but JSON uses `null` and retains the raw handle in
`details.function_handle`. The contents of `details` vary by API. Trace output
can contain sensitive source paths, function names, handles, and timing data;
treat saved JSONL files accordingly.

### Capture statistics and summary

`--stats` prints one final diagnostic line to standard error without changing
table or JSONL event output:

```text
metagross: stats events=541 attributed=276 unknown=265 errors=0 lost=0 dropped=0 complete=true
```

`--summary-output FILE` writes a separate, versioned JSON document with capture
completeness, API/error counts, CPU-side duration totals, successful copy bytes,
observed allocation totals, and the top attributed functions and resolved
kernels:

```sh
sudo /usr/bin/python3 -m metagross \
  --json --output /tmp/events.jsonl \
  --stats --summary-output /tmp/summary.json \
  examples/gpu_demo.py
```

Summary schema version 1 has top-level `schema_version`, `complete`, `capture`,
`timing`, `memory`, `copies`, `apis`, `top_functions`, `top_kernels`,
`configuration`, and `target` fields. `complete` is false if BPF events were
lost, nested calls were dropped, event rendering failed, or the tracing loop
failed. Unknown Python attribution does not by itself make capture incomplete.
Allocation and byte totals describe successfully observed driver calls, not
physical GPU usage or framework-level tensor allocations.

### Visual trace snapshot

The unprivileged viewer streams an existing JSONL trace into a bounded in-memory
model and prints a terminal dashboard. It does not import BCC, inspect libcuda,
or require root, CUDA, or a GPU:

```sh
/usr/bin/python3 -m metagross view \
  --snapshot --summary /tmp/summary.json \
  /tmp/events.jsonl
```

The snapshot shows capture health, attribution coverage, CUDA errors, CPU API and
synchronization duration, copy and observed-memory totals, top APIs, functions,
kernels, and recent events. Use `--width COLUMNS` to make output deterministic
for CI or saved reports and `--recent N` to bound retained recent events. The
viewer tolerates malformed lines when valid records remain, reports their count,
sanitizes terminal control characters, and warns when summary and event counts
differ. Interactive curses and live-follow modes are planned next.

## Traced API table

Metagross observes the following CUDA driver API families:

| Family | APIs |
|--------|------|
| Kernel launches | `cuLaunchKernel`, `cuLaunchKernelEx` |
| Memory management | `cuMemAlloc`, `cuMemAllocAsync`, `cuMemFree`, `cuMemFreeAsync` |
| Memory transfers | `cuMemcpyHtoD` (host to device), `cuMemcpyDtoH` (device to host), `cuMemcpyDtoD` (device to device), `cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyDtoDAsync`, `cuMemcpy`, `cuMemcpyAsync` |
| Synchronization | `cuStreamSynchronize`, `cuCtxSynchronize`, `cuEventSynchronize` |
| Internal name registration (not rendered as rows) | `cuModuleGetFunction`, `cuLibraryGetKernel`, `cuKernelGetFunction` |

Details captured depend on the API:

| API | Details |
|-----|---------|
| Kernel launches | `grid`, `block`, `shared`, `stream`, `function_handle` |
| Allocations | `bytes`, `ptr`, `stream` (async only), `gpu_total` |
| Deallocations | `ptr`, `gpu_total`, plus `bytes` when the pointer is known and `stream` for async calls |
| Memory transfers | `bytes`, `stream` (async only) |
| Synchronization | `stream` (cuStreamSynchronize), `event` (cuEventSynchronize) |

Versioned and per-thread-default-stream symbol variants are normalized to the
base API names above. For directional transfers, direction is encoded in the
`api` field: `cuMemcpyHtoD` is host-to-device and `cuMemcpyDtoH` is
device-to-host. Generic `cuMemcpy` and `cuMemcpyAsync` events remain generic.
`gpu_total` is the running total of successfully observed driver allocations;
it is not a measurement of all memory owned by a framework or process.

## Overhead and limits

The profiling callback runs for every Python call/return and emits records for
project frames. Short-lived or Python-call-heavy programs can therefore slow
down substantially; measure overhead on the target workload. Metagross is
intended for local diagnosis rather than production monitoring and traces only
the main Python process.

**CPU-side timing only**: Kernel launches are asynchronous and return immediately
to the caller. Reported durations measure the API call's lifetime in CPU time, not
actual kernel execution. Real kernel stalls appear in synchronization call
durations like `cuStreamSynchronize`.

**Kernel names best-effort**: Kernel function names are available only if the
target calls `cuModuleGetFunction`, `cuLibraryGetKernel`, or `cuKernelGetFunction`
to register the kernel before launch. Unresolved handles use the table/JSON
behavior described in [Output](#output).

**PyTorch autograd**: Some backward-pass kernels may report `<unknown>` because
PyTorch can launch them from C++ worker threads without an active Python project
frame.

**PyTorch caching allocator**: PyTorch's GPU memory caching allocator reserves
and reuses driver allocations. After warmup, driver-level allocations can become
rare even while tensor allocation continues. Reported allocation counts reflect
driver activity, not logical tensor allocations.

**VMM and expandable segments**: Virtual Memory Management (VMM) and expandable
segments allocations are invisible to the driver API tracing: they do not appear
as `cuMemAlloc` calls.

**Nested re-entry**: Nested driver-API re-entry (rare) drops the outer event,
keeping only the innermost call. Such events are counted but not duplicated.

**High event rates**: A workload that emits CUDA calls faster than userspace can
drain and render them can overflow the BPF ring buffer. Metagross prints a lost
event warning; a trace with that warning is incomplete.

The Docker example includes a repeatable bare/full/no-attribution/launch-only
[overhead comparison](examples/docker/README.md#benchmark-tracing-overhead).

It does not attach to an existing PID or run modules with `-m`. It does not
identify async tasks or individual source lines. A C extension API call can still
be attributed to its nearest active project Python caller.

Target exit statuses from 0 through 255 are preserved. A target signal returns
`128 + signal`, including 130 for Ctrl-C. Metagross returns 1 for validation,
dependency, privilege, probe, compile, attach, transport, or cleanup
failures, and 2 for invalid command-line syntax. A broken trace output stops
rendering but lets the target finish and preserves its status.

## Verification

Run the unprivileged suite without BCC:

```sh
/usr/bin/python3 -m unittest -v
```

After installing BCC and driver, run the host-kernel integration test as root:

```sh
sudo env RUN_EBPF_INTEGRATION=1 \
  /usr/bin/python3 -m unittest -v test_metagross.LiveTraceTest
```

The integration gate exercises kernel launches, memory operations,
synchronization, process credentials, and eBPF cleanup. When `bpftool` is
available, it also checks that no additional BPF program remains after shutdown.
The containerized PyTorch workloads have a separate GPU correctness suite; see
[`examples/docker/README.md`](examples/docker/README.md#run-the-workload-correctness-tests).
