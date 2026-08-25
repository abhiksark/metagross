# Metagross: eBPF GPU-call tracing for one Python script

Metagross runs one local Python script and explains which project function caused
each selected CUDA driver API call. The target keeps its normal standard output and
standard error. Trace records use a readable table by default or JSONL for
machine processing.

The implementation targets Ubuntu 22.04, Linux 6.8, x86_64, and
`/usr/bin/python3` with NVIDIA CUDA driver installed.

## Setup

Install the Ubuntu BCC Python package and ensure the NVIDIA driver is installed:

```sh
sudo apt install python3-bpfcc
```

The NVIDIA driver is required for live GPU tracing. Help, imports, source
inspection, and unit tests work without it. BCC and root privileges are required
only for live tracing. Upstream provides the
[BCC installation guide](https://github.com/iovisor/bcc/blob/master/INSTALL.md)
and [Python API source](https://github.com/iovisor/bcc/blob/master/src/python/bcc/__init__.py).

## Usage

Run the included demonstration from the repository root:

```sh
sudo /usr/bin/python3 -m metagross examples/gpu_demo.py
```

The complete interface is:

```text
sudo /usr/bin/python3 -m metagross \
  [--json] [--output FILE] [--project-root DIR] \
  script.py [script arguments...]
```

Metagross options must appear before the script. Everything after the script is
passed to it unchanged. `--project-root` defaults to the current directory.
The script must resolve to a regular `.py` file inside that directory.

Without `--output`, trace records go to standard error so the target's standard
output remains unchanged. Diagnostics and target standard error can share that
stream. Use a separate file for reliable machine processing:

```sh
sudo /usr/bin/python3 -m metagross \
  --json --output /tmp/metagross.jsonl \
  examples/gpu_demo.py
```

New output files have mode `0600` and are owned by the invoking user. Metagross
truncates an existing output only when it is a regular, non-symlink file owned
by that user.

Inspect generated eBPF C code without importing BCC or requiring root:

```sh
/usr/bin/python3 -m metagross --ebpf
```

## How attribution works

The child process uses Python profiling hooks to report project function calls
and returns. One eBPF program reports selected CUDA driver API entries and
another reports their completion. Metagross joins the streams by native thread ID
and the shared monotonic clock, then attributes each API call to the project
frame that was active at API entry.

Calls in the standard library, site packages, virtual-environment packages,
and Metagross itself do not replace the nearest project frame. A memory allocation
performed inside a standard-library function, for example, remains attributable to
the project function that initiated it. Imported project modules and Python
threads are included.

The target remains behind a pipe barrier until both raw API tracepoints are
attached for its exact process ID. Startup API calls before the first project
frame are suppressed. Events are held for 100 ms to tolerate cross-CPU and
cross-stream delivery ordering. Long API calls appear after they complete.

## Output

The table begins with this shape:

```text
TIME         FUNCTION    LOCATION    API          RET  DURATION  DETAILS
12:10:03.41  compute     gpu_demo.py:86  LaunchKernel 0  0.05ms    grid=8,1,1 block=128,1,1 kernel=vec_add stream=0x0
```

Detail strings use shell-safe quoting. `<unknown>` means the profile or eBPF
stream did not contain enough matching data for safe attribution.

JSONL records use this exact top-level schema:

```json
{"timestamp":"2026-08-24T12:10:03.410000+05:30","pid":1234,"tid":1234,"function":"compute","file":"gpu_demo.py","line":86,"api":"cuLaunchKernel","kernel":"vec_add","return_code":0,"duration_ns":50000,"details":{"grid":"8,1,1","block":"128,1,1","stream":"0x0","function_handle":"0xf00"}}
```

`function`, `file`, and `line` are `null` when attribution is unknown. `line` is
the function definition line. `return_code` is the signed raw kernel result.
Timestamps are local ISO 8601 times derived from the API call's monotonic start
time and the parent startup wall-clock offset.

## Traced API table

Metagross reports on the following CUDA driver API families:

| Family | APIs |
|--------|------|
| Kernel launches | `cuLaunchKernel`, `cuLaunchKernelEx` |
| Memory allocation | `cuMemAlloc`, `cuMemAllocAsync`, `cuMemFree`, `cuMemFreeAsync` |
| Memory transfers | `cuMemcpyHtoD` (host to device), `cuMemcpyDtoH` (device to host), `cuMemcpyDtoD` (device to device), `cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyDtoDAsync`, `cuMemcpy`, `cuMemcpyAsync` |
| Synchronization | `cuStreamSynchronize`, `cuCtxSynchronize`, `cuEventSynchronize` |
| Name registration | `cuModuleGetFunction`, `cuLibraryGetKernel`, `cuKernelGetFunction` |

Details captured depend on the API:

| API | Details |
|-----|---------|
| Kernel launches | `grid`, `block`, `shared`, `stream`, `function_handle` |
| Allocations | `bytes`, `ptr`, `stream` (async only), `gpu_total` |
| Deallocations | `bytes`, `ptr`, `stream` (async only), `gpu_total` |
| Memory transfers | `bytes`, `stream` (async only) |
| Synchronization | `stream` (cuStreamSynchronize), `event` (cuEventSynchronize) |

For memory transfers, direction is encoded in the `api` field: `cuMemcpyHtoD`
indicates host-to-device, `cuMemcpyDtoH` indicates device-to-host.

## Overhead and limits

Python profiling reports every project call and return, so Metagross is intended
for local diagnosis rather than production monitoring. It traces only the main
Python process.

**CPU-side timing only**: Kernel launches are asynchronous and return immediately
to the caller. Reported durations measure the API call's lifetime in CPU time, not
actual kernel execution. Real kernel stalls appear in synchronization call
durations like `cuStreamSynchronize`.

**Kernel names best-effort**: Kernel function names are available only if the
target calls `cuModuleGetFunction`, `cuLibraryGetKernel`, or `cuKernelGetFunction`
to register the kernel before launch. If the kernel handle is not registered,
the name field reports `kernel@0x...` (the handle in hexadecimal).

**PyTorch autograd**: PyTorch's autograd backward kernels report `<unknown>` for
the function name because the backward pass runs in C++ without Python frame
attribution.

**PyTorch caching allocator**: PyTorch's GPU memory caching allocator manages a
large pre-allocated pool. After warmup, driver-level allocations become rare as
PyTorch reuses pre-allocated memory. Reported allocation counts reflect this
pooling behavior.

**VMM and expandable segments**: Virtual Memory Management (VMM) and expandable
segments allocations are invisible to the driver API tracing: they do not appear
as `cuMemAlloc` calls.

**Nested re-entry**: Nested driver-API re-entry (rare) drops the outer event,
keeping only the innermost call. Such events are counted but not duplicated.

It does not attach to an existing PID or run modules with `-m`. It does not
identify async tasks or individual source lines. A C extension API call can still
be attributed to its nearest active project Python caller.

Target exit statuses from 0 through 255 are preserved. A target signal returns
`128 + signal`, including 130 for Ctrl-C. Metagross returns 1 for validation,
dependency, privilege, tracepoint, compile, attach, transport, or cleanup
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

The integration gate exercises kernel launches, memory operations, synchronization,
process credentials, and eBPF cleanup. When `bpftool` is available, it also checks
that no additional BPF program remains after shutdown.
