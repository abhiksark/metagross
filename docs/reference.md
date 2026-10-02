# Metagross reference

This reference describes the source-only experimental preview. Start with the
[quick start](../README.md#quick-start) for a first local trace. For development
and verification gates, see [Contributing](../CONTRIBUTING.md).

## Usage

Run the included demonstration from the repository root:

```sh
sudo /usr/bin/python3 -m metagross examples/gpu_demo.py
```

Metagross can also be installed with pip from a checkout, which adds a
`metagross` command and makes `import metagross` work from any directory:

```sh
/usr/bin/python3 -m pip install .
metagross --version
```

Install it into the Python that has the BCC bindings if you want to trace with
it; pip cannot install BCC. The installed command takes the same arguments as
`python3 -m metagross` and never has the working directory on its import path.
It has the same name as the Docker wrapper from the quick start, so whichever
comes first on `PATH` runs.

### Run without the Docker image

The script runs under the interpreter that started Metagross, and that
interpreter must be able to import BCC. Distributions package the BCC bindings
for their system Python only (`python3-bpfcc` on Ubuntu) and pip cannot
install them, so everything the script imports has to be importable from an
interpreter that also sees the system BCC:

- **System Python**: install the script's packages for `/usr/bin/python3` and
  start Metagross with it, as in the examples in this reference.
- **Virtual environment**: create it from the system Python with access to the
  system packages, and start Metagross with the environment's interpreter:

  ```sh
  /usr/bin/python3 -m venv --system-site-packages .venv
  .venv/bin/pip install your-packages
  sudo .venv/bin/python /path/to/metagross-checkout/metagross script.py
  ```

  The script then runs inside the environment. An environment created
  without `--system-site-packages`, or a conda or pyenv interpreter, cannot
  import the system BCC and stops with `bcc (BPF Compiler Collection) not
  available`.
- **Your own image**: follow
  [`examples/docker/Dockerfile`](../examples/docker/Dockerfile). Install
  `python3-bpfcc` and your packages for the image's system Python, copy the
  `metagross` package in, set `SUDO_UID`, `SUDO_GID`, and `SUDO_USER` to the
  account the script should run as, and start the container with the flags the
  [Docker guide](../examples/docker/README.md#run-the-basic-workload)
  explains. The distribution's system Python decides the Python version:
  3.10 on Ubuntu 22.04, 3.12 on Ubuntu 24.04.

Rows come from any code in the process that calls the traced driver APIs in
the probed `libcuda.so.1`. PyTorch and direct driver calls through `ctypes`
are tested; other libraries, such as CuPy, ONNX Runtime, and TensorRT, are
not.

The tracing interface is:

```text
metagross \
  [--json] [--output FILE] [--stats] [--summary-output FILE] \
  [--project-root DIR] [--trace FAMILIES] [--no-attribution] \
  [--allow-root-target] \
  [--dashboard-port PORT | --web [--web-port PORT]] \
  script.py [script arguments...]
```

Here `metagross` stands for whichever way you start it: the installed
command, `/usr/bin/python3 -m metagross` from the checkout, or the
[path form](#privilege-and-trust-boundary). Tracing a script needs root.

Metagross options must appear before the script. Everything after the script is
passed to it unchanged. An option that takes a value accepts `--name value`
and `--name=value`. A script whose name starts with a dash is written as
`./-name.py`. `--project-root` defaults to the current directory.
The script must resolve to a regular `.py` file inside that directory.
Use `/usr/bin/python3 -m metagross --help` (or `-h`) for usage and `--version`
for the version, without root, BCC, CUDA, or a target script. Help flags after
the script go to the target.

| Tracer option | Default and behavior |
|---------------|----------------------|
| `--json` | Off; emit JSONL instead of human-readable table rows. |
| `--output FILE` | No file; trace records go to stderr. |
| `--stats` | Off; print a final capture statistics line to stderr. |
| `--summary-output FILE` | No file; write a separate version-1 JSON summary when selected. |
| `--project-root DIR` | Current directory; project attribution and target validation boundary. |
| `--trace FAMILIES` | `all`; comma-separated `launch`, `memory`, `copy`, `sync`. |
| `--no-attribution` | Off; disable the Python profile hook. |
| `--allow-root-target` | Off; when there is no sudo caller to drop to, run the script as root instead of refusing. |
| `--dashboard-port PORT` | Disabled; integer 1 to 65,535 for direct loopback delivery to a separately started receiver. |
| `--web` | Off; start a [built-in dashboard](#built-in-web-dashboard) for this capture. Rejects `--dashboard-port`. |
| `--web-port PORT` | 8765; built-in dashboard port, 0 to 65,535, where zero selects a free port. Requires `--web`. |
| `--ebpf` | Print generated C without tracing; accepts `--trace`, rejects `--dashboard-port` and `--web`. |
| `-h`, `--help` | Print usage without a target or privileged dependencies. |
| `--version` | Print the version without a target or privileged dependencies. |

`--trace all` cannot be combined with another family. Empty or unknown families
are rejected. A producer token in the environment alone does not enable delivery.

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
file owned by that user with no other hard link, verified through the opened
file descriptor. Trace
and summary outputs must refer to different files, including through hard links;
both are validated before either existing file is truncated. An existing file
is emptied only once the probes are attached and the script is about to start,
so a run that fails to start leaves the earlier capture intact.

Output parent directories must be owned by root or the invoking user. Symlinked
parents and group- or other-writable non-sticky parents are rejected. Standard
sticky directories such as `/tmp` are supported. The check covers every
directory from `/` down to the output file, so a private directory inside a
group-writable one is still refused; the error names the directory to fix.
The directory that holds the file must belong to the invoking user or be a
shared sticky directory, so that Metagross, running as root, never creates a
file for you where you could not have created it yourself. Ownership changes apply only
to newly created open files, never to a replacement pathname. Another process
with permission to rename your files can still move or unlink the capture;
keep its directory private when stable paths matter.

Under sudo, the controller retains the privileges needed for BPF while the
target runs as the validated invoking user. Without a sudo caller to drop to,
Metagross refuses to start; `--allow-root-target` runs the script as root and
emits a warning.
After probes attach, the child enters a fresh Python interpreter with the same
PID, preserving interpreter options such as `-O`/`-OO`, `-B`, `-u`, `-W`, and `-X utf8`.
Normal interpreter shutdown waits for non-daemon threads, runs `atexit`
handlers, and flushes buffered target output.

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
frame its calling thread was in. A thread stays inside the driver for the
whole call, so its frame at entry and at return are the same; Metagross looks
it up at return, which is why a call that blocks for seconds keeps its frame.

Calls in the standard library, site packages, virtual-environment packages,
Metagross itself, and code with no source file (`exec`, frozen modules,
generated code) do not replace the nearest project frame. A memory
allocation performed inside a standard-library function, for example, remains
attributable to the project function that initiated it. Imported project
modules and threads started through `threading` are included; threads started
with the low-level `_thread` module are not profiled, so their calls are
`<unknown>`.

Attribution follows the interpreter, not object lifetime. Python reports a
function's return before it releases that function's local variables, so a
driver call made by a destructor at function exit, such as a free, is
attributed to the caller.

The target remains behind a pipe barrier until every uprobe and uretprobe
pair is attached for its exact process ID. API calls that occur before any
project frame is active are attributed as unknown rather than suppressed.
A call is written once it is at least 100 ms old and Metagross has read the
profile stream past the call's entry: a newer profile record has arrived, or
the stream was empty afterwards. Until then a record may still be on its way,
and the frame seen so far would be a guess. If the stream has still not reached
the call five seconds after it began, the call is written as `<unknown>` and
counted in `refused_attributions`. Long API calls appear after they complete,
attributed to the frame that made them, so rows are not in strict timestamp
order; sort by `timestamp` if the order matters.

### Op spans

`metagross.span("name")` is a public context manager the target script can
call to mark a named region of its own code. It and `metagross.__version__`
are the whole Python API; every other name in the package is internal and can
change in any release.

```python
import metagross

with metagross.span("forward"):
    model(inputs)
```

Every CUDA API call made while a span is open carries that span's name in its
`span` field, alongside its usual project function attribution. Spans nest:
the innermost open span is the one recorded.

Each thread and each `asyncio` task has its own spans. A span held open across
an `await` labels only the calls its own task makes; other tasks that run on
the same thread in the meantime keep their own span, or `null`. A task or
thread started inside a span (for example with `asyncio.create_task` or
`asyncio.to_thread`) is labelled with it only while the span stays open.
Threads started any other way, including `loop.run_in_executor` workers,
start with no span.

A `metagross.span()` call outside a running trace (the script run bare, without
`sudo /usr/bin/python3 -m metagross`) is a no-op: the `with` block still
runs its body normally, so instrumented scripts stay runnable unmodified as long
as `import metagross` works there (install it with pip, or put the checkout on
`PYTHONPATH`).
Spans use the same profile pipe and interning writer as project-function
attribution (see [Overhead and limits](#overhead-and-limits)) and fail closed
rather than guess:

- After a lost profile record, a thread's span is `null` until its span next
  changes.
- Closing a span while another span opened later in the same task or thread is
  still open (possible with interleaved generators or manual `__enter__` and
  `__exit__` calls) makes every span open there report `null` until it closes.
- Task switches are detected when the resumed task enters a Python function,
  which `asyncio` always does. Schedulers that resume in the middle of a
  function, such as greenlets and gevent, are not supported: their calls can
  carry another greenlet's span.

Once a script opens its first span, the profiling hook compares the running
task's span with the last one reported on every Python call, which adds about
0.1 µs per call. Scripts that never open a span do not pay this.

## Output

The table begins with this illustrative shape (timings vary):

```text
TIME        FUNCTION          LOCATION            API               RET  DURATION DETAILS
12:10:03.41 compute           gpu_demo.py:86      LaunchKernel      0    50.0us   kernel=vec_add grid=8,1,1 block=128,1,1 shared=0 stream=0x0
```

`DURATION` uses the unit that fits the value: `ns`, `us`, `ms`, or `s`.

Detail strings use shell-safe quoting. Control characters in a row, which can
only come from target-supplied kernel, span, or file names, are printed as `?`.
`<unknown>` means the profile or eBPF
stream did not contain enough matching data for safe attribution.

JSONL records use this exact top-level schema:

```json
{"timestamp":"2026-08-24T12:10:03.410000+05:30","pid":1234,"tid":1234,"function":"compute","file":"/workspace/examples/gpu_demo.py","line":86,"api":"cuLaunchKernel","kernel":"vec_add","return_code":0,"duration_ns":50000,"details":{"grid":"8,1,1","block":"128,1,1","shared":0,"stream":"0x0","function_handle":"0xf00"},"span":"compute"}
```

`function`, `file`, and `line` are `null` when attribution is unknown. `file` is
the path reported by the Python code object and is normally absolute; table
output displays only its basename. `line` is the function definition line.
`return_code` is the signed raw CUDA driver `CUresult`. Timestamps are local ISO
8601 values derived from the API call's monotonic start time and the parent
startup wall-clock offset. `span` is the name of the innermost
[`metagross.span()`](#op-spans) region active at API entry in the calling thread
or `asyncio` task, or `null` when no span was active; it is additive and always
present.

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
metagross: stats events=541 attributed=276 unknown=265 errors=0 lost=0 dropped=0 lost_profile=0 refused=0 complete=true
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
`top_spans`, `configuration`, and `target` fields. `complete` is false if BPF
events were lost, nested calls were dropped, profile records were lost, the
target replaced the profiling hook, the script ended without a normal
interpreter exit, Metagross left calls unattributed because it lacked their
profile history, the script loaded a `libcuda` the probes are not on, event
rendering failed, or the tracing loop failed.
`lost_profile_records` counts one for each hook replacement and one for a
profile stream that stopped before the script's normal exit.
`refused_attributions` counts calls written as `<unknown>` although their
thread may have been inside a project function: the profile history for that
moment had been lost, or had not arrived within five seconds.
Unknown attribution for any other reason, such as a call from a thread with no
project frame, does not make capture incomplete. Allocation and byte totals describe
successfully observed driver calls, not physical GPU usage or
framework-level tensor allocations.

Summary nested fields are:

| Object | Fields |
|--------|--------|
| `capture` | `events`, `attributed`, `unknown_attribution`, `cuda_errors`, `lost_events`, `dropped_nested_calls`, `render_failed`, `trace_failed`, `lost_profile_records`, `refused_attributions`, `libcuda_mismatch` |
| `timing` | `total_api_duration_ns`, `synchronization_duration_ns` |
| `memory` | `successful_allocation_bytes`, `observed_peak_bytes`, `observed_outstanding_bytes` |
| `copies` | `successful_bytes_by_api`, mapping normalized API names to byte counts |
| `configuration` | `trace_families`, `python_attribution`, `attached_symbol_variants`, `attached_probes` |
| `target` | `pid`, `script`, `exit_status` |

The `apis` array identifies rows by `api`; `top_functions` identifies rows by
`function`, `file`, and `line`; `top_kernels` identifies rows by `kernel`;
`top_spans` identifies rows by `span`, mirroring `top_kernels`, and is
populated only from events attributed to a `metagross.span()` region.
Every aggregate row contains `count`, `errors`, `total_duration_ns`,
`max_duration_ns`, and `successful_bytes`. Function, kernel, and span lists
retain the top 20 groups ordered by count, then total duration, then
identity. Counts and durations are integers; failure indicators and
`complete` are booleans.
The [sanitized summary fixture](../examples/captures/basic-summary.json) is a
complete example. `capture.delivery_dropped` exists only in direct-delivery
summaries, as described below.

### Visual trace viewer

The unprivileged viewer accepts saved JSONL files and authenticated direct
captures. It does not import BCC, inspect libcuda, or require root, CUDA, or a
GPU. Print a static terminal dashboard from a completed trace with:

```sh
/usr/bin/python3 -m metagross view \
  --snapshot --summary /tmp/summary.json \
  /tmp/events.jsonl
```

`--width COLUMNS` is snapshot-only, accepts 60 through 240, and makes output
deterministic for CI or saved reports. `--recent N` bounds retained recent
events in every viewer mode (maximum 10,000).

To watch a trace while its target is running, start the dependency-free curses
dashboard in an interactive terminal. The trace may not exist yet; the viewer
waits for its creation:

```sh
/usr/bin/python3 -m metagross view \
  --follow --summary /tmp/summary.json \
  /tmp/events.jsonl
```

The live overview refreshes every 0.2 seconds and shows event rate, attribution,
CUDA errors, CPU API and synchronization duration, copies, observed memory, top
APIs/functions/kernels, and recent events. Press `p` to pause file consumption
and `q` to exit. Both live dashboards honor `--refresh SECONDS`, which accepts
values from 0.05 to 5.0. Follow mode requires an interactive terminal of at
least 80 columns by 18 rows on both stdin and stdout; use `--snapshot` for
redirected terminal output.

Serve the same live model as a responsive browser dashboard:

```sh
/usr/bin/python3 -m metagross view \
  --web --summary /tmp/summary.json \
  /tmp/events.jsonl
```

Open the complete private URL printed in the terminal, including its
`#viewer_token=...` fragment. The browser saves this viewer credential in
session storage and removes the fragment from the visible URL before requesting
trace data. Refreshing the same tab keeps access; after restarting the server,
open its newly printed URL. Session storage must be enabled.

Session storage belongs to one tab, and the address bar keeps only the URL
without the fragment, so a new tab or a history suggestion opens the page
without a credential. The page then shows a field that accepts the viewer token
or the complete private URL. A pasted value is stored the same way and never
enters the address bar or browser history.

By default the server listens only on the local loopback interface. To view the
dashboard from another machine on the internal network, add `--host` with one of
this machine's internal IPv4 addresses, which then appears in the printed URL,
or `--host 0.0.0.0` for all interfaces, in which case replace `<this-host-ip>`
in the printed URL with one of this machine's internal addresses. The browser
must address the server by IPv4 address or `localhost`; other hostnames are
rejected.

This mode is internal-network only. `--host` accepts `127.0.0.1`, `0.0.0.0`,
and private, link-local, or shared-range (for example Tailscale) IPv4 addresses,
and refuses public ones. On a carrier-grade NAT network, the shared range can
also include other subscribers. The server also refuses requests from public-internet
client addresses. These checks add defense in depth; they do not replace a
firewall. A non-loopback bind prints a warning: the dashboard uses plain HTTP,
so the viewer token and trace data cross the network unencrypted, and anyone
with the URL can read trace source paths and timing. Capture ingest from the
tracer stays loopback-only, and `--receive` accepts only `--host 127.0.0.1` or
`--host 0.0.0.0` because the tracer delivers captures only to `127.0.0.1`.

The server reads the trace and summary in a bounded background follower, and
needs no TTY, root, BCC, CUDA, GPU, JavaScript packages, or external network
access. Choose another port with `--port PORT`; use `--port 0` to let the OS
select a free one. The timeline-first workspace groups CUDA driver
API events by attributed project function and provides API/function/kernel
and span search, API-family filters, 1x to 16x zoom, drag-to-pan navigation, synchronized
timeline and event-table selection, and a source/detail inspector. Compute-style
summary sections retain allocation history and top APIs, functions, and kernels.
Clipped timeline labels expose their full event summary on pointer hover or
keyboard focus. Events View rows support Enter and Space selection; narrow
layouts keep report and refresh controls visible while containing timeline and
table scrolling within their panels.
Timeline bars show observed CPU-side CUDA API call duration; they do not claim
GPU kernel execution timing. Stop the server with Ctrl-C.

The dashboard requires its independent viewer bearer token for `/api/state`.
HTML, CSS, JavaScript, and image assets are readable without authentication and
contain no trace data or embedded credentials. The producer secret in
`METAGROSS_DASHBOARD_TOKEN` authorizes only capture POST requests and is never
sent to the browser; a viewer credential cannot publish captures.

Treat the private URL as access to sensitive trace paths, function names, and
timings. The default loopback binding limits network exposure; it does not
authenticate other local users. With `--host`, the dashboard is also reachable
from the internal network over plain HTTP; use it only on a trusted internal
network.
Viewer authentication blocks callers without the secret, but does not protect
against root, a compromised user account or browser, or software able to read
the terminal output or browser session storage. Do not share the private URL,
and do not expose the server through a reverse proxy or a tunnel that other
people can reach. Use `--host` when viewers on the internal network need
access, or an SSH forward to your own machine as described under
[Built-in web dashboard](#built-in-web-dashboard).

The live status moves from `WAITING` to `LIVE`, then reconciles to `COMPLETE`,
`INCOMPLETE`, or `MISMATCH` when the final summary appears. An empty summary
file is pending while capture runs; a non-empty invalid summary produces a
visible warning and is retried when it changes. Valid traces containing malformed
lines retain the final status with a ` / MALFORMED` suffix.

All viewer modes tolerate malformed lines when valid records remain, report
their count, sanitize control characters, and bound line, summary, aggregate,
and recent-event storage. The live terminal and browser modes hold partial JSONL
records until their newline arrives and reset safely when the trace is truncated
or replaced.

## Traced API table

Metagross observes the following CUDA driver API families:

| Family | APIs |
|--------|------|
| Kernel launches | `cuLaunchKernel`, `cuLaunchKernelEx`, `cuGraphLaunch` |
| Memory management | `cuMemAlloc`, `cuMemAllocAsync`, `cuMemFree`, `cuMemFreeAsync` |
| Memory transfers | `cuMemcpyHtoD` (host to device), `cuMemcpyDtoH` (device to host), `cuMemcpyDtoD` (device to device), `cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyDtoDAsync`, `cuMemcpy`, `cuMemcpyAsync` |
| Synchronization | `cuStreamSynchronize`, `cuCtxSynchronize`, `cuEventSynchronize` |
| Internal name registration (not rendered as rows) | `cuModuleGetFunction`, `cuLibraryGetKernel`, `cuKernelGetFunction` |

Details captured depend on the API:

| API | Details |
|-----|---------|
| Kernel launches | `grid`, `block`, `shared`, `stream`, `function_handle` |
| Graph launches | `graph_exec`, `stream` |
| Allocations | `bytes`, `ptr`, `stream` (async only), `gpu_total` |
| Deallocations | `ptr`, `gpu_total`, plus `bytes` when the pointer is known and `stream` for async calls |
| Memory transfers | `bytes`, `stream` (async only) |
| Synchronization | `stream` (cuStreamSynchronize), `event` (cuEventSynchronize) |

Versioned and per-thread-default-stream symbol variants are normalized to the
base API names above. JSONL retains these names; table rows omit the `cu` prefix,
for example `LaunchKernel` and `MemAlloc`. For directional transfers, direction is encoded in the
`api` field: `cuMemcpyHtoD` is host-to-device and `cuMemcpyDtoH` is
device-to-host. Generic `cuMemcpy` and `cuMemcpyAsync` events remain generic.
`gpu_total` is the running total of successfully observed driver allocations;
it is not a measurement of all memory owned by a framework or process.

## Overhead and limits

The profiling callback runs for every Python call/return and emits records for
project frames. On one desktop machine it added roughly 4 to 5 µs to each call
of a project function and 0.4 to 0.5 µs to every other Python call (Python 3.10
and 3.13). Short-lived or Python-call-heavy programs can therefore slow down
substantially; measure overhead on the target workload. On the same machine a
traced driver call took about 2 µs longer at the median and 3 µs at the 99th
percentile: a PyTorch kernel launch went from 2.8 µs to 4.7 µs with the launch
probes alone, and to 5.2 µs with every probe and Python attribution. Metagross
is intended for local diagnosis rather than production monitoring and traces
only the main Python process.

**Host-side timing only**: Reported durations are elapsed monotonic time from
CUDA API entry to return, including waiting and time when the calling thread is
descheduled. They do not measure CPU execution time or GPU kernel execution time.
Asynchronous submission does not guarantee an immediate return: CUDA API calls
may block for internal-resource reasons. Synchronization durations show how long
the host call took to return, but do not identify individual kernel execution
times.

**Kernel names best-effort**: Kernel function names are available only if the
target calls `cuModuleGetFunction`, `cuLibraryGetKernel`, or `cuKernelGetFunction`
to register the kernel before launch. Unresolved handles use the table/JSON
behavior described in [Output](#output). Names are C++ mangled symbols and can
be several hundred characters long. JSONL and the summary keep up to 1,023
bytes of a name; table rows show the first 124 characters followed by `...`.

**CUDA graphs**: Replaying a graph is one `cuGraphLaunch` row with no kernel
name; the kernels inside the graph are not listed, because the driver replays
them without calling `cuLaunchKernel`. While a graph is being captured from a
stream, each kernel recorded into it appears as an ordinary launch row,
although it only runs when the graph is replayed. A graph built node by node
with `cuGraphAddKernelNode` produces no launch rows, only its replays.

**Untraced driver APIs**: Only the APIs in the [table](#traced-api-table) are
traced. In particular, `cuMemsetD*`, 2D, 3D, peer and batched copies,
pooled, managed, host and pitched allocations, `cuStreamWaitEvent`,
`cuLaunchCooperativeKernel`, and `cuLaunchHostFunc` are not
recorded. A capture of a workload that relies on these is partial even when it
reports `complete`.

**Calls in flight at exit**: An API call is reported when it returns. A call
that is still running when the target exits or is killed never appears.

**Abnormal exit**: If the script is killed by a signal or leaves through
`os._exit`, Metagross cannot confirm that its last profile records arrived.
The capture is marked incomplete and the calls not yet printed, about the last
100 ms, are `<unknown>`. An exception, `sys.exit`, or Ctrl-C ends the script
normally.

**No device or context**: Events do not record which GPU or CUDA context a call
used, so multi-GPU activity is not separated.

**x86-64 only**: Launch arguments are read from the System V AMD64 stack
layout and libcuda is looked up in x86-64 library paths. Other architectures,
including arm64, are not supported, and Metagross refuses to trace on them.

**One libcuda**: Probes are attached to the `libcuda.so.1` Metagross finds
in the standard library paths or the loader cache. If the script loads a
different copy, for example through `LD_LIBRARY_PATH`, none of its calls are
traced. Metagross checks the script's loaded libraries ten times a second
until a `libcuda` appears; if it is a different file, it warns on stderr and
marks the capture incomplete (`libcuda_mismatch`). A script that exits before
the first check is not checked.

**Host PID namespace**: The probes select the script by the process ID the
kernel reports, which is its ID in the host PID namespace. Metagross refuses
to start in any other PID namespace, for example a container started without
`--pid=host`, because no call would be matched.

**Other profilers**: Attribution uses the Python profiling hook
(`sys.setprofile`). If the script installs its own profile function, as
`cProfile` does on Python 3.11 and earlier, Metagross forgets the frames open at
that moment, prints a warning, and marks the capture incomplete. Calls on that
thread report `<unknown>` until the script restores the hook, which `cProfile`
does not do. On Python 3.12 and later `cProfile` runs alongside the hook and
attribution is unaffected. A replacement made from a thread that Metagross does
not profile is not detected. `threading.setprofile()` changes the hook that
later threads start with and is noticed only when the script exits: calls on
those threads are `<unknown>` and the capture is marked incomplete.

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

**Driver-internal calls**: The driver can call its own traced entry points,
for example to allocate memory on behalf of another API. Such a call is an
ordinary row, attributed to the project function that was active, and it
counts toward `gpu_total`.

**High event rates**: A workload that emits CUDA calls faster than userspace can
drain and render them can overflow the BPF ring buffer. Metagross prints a lost
event warning; a trace with that warning is incomplete.

**High Python call rates**: A script that calls project functions faster than
Metagross reads the records, several hundred thousand calls per second on one
desktop machine, fills the profile pipe. The script then drops records instead
of waiting. Metagross forgets the frames it knew at that point, so calls near
the loss are `<unknown>`, never a stale frame, and the capture is incomplete.

The Docker example includes a repeatable bare/full/no-attribution/launch-only
[overhead comparison](../examples/docker/README.md#benchmark-tracing-overhead).

It does not attach to an existing PID or run modules with `-m`. It does not
identify async tasks or individual source lines. A C extension API call can still
be attributed to its nearest active project Python caller.

Target exit statuses from 0 through 255 are preserved. A target signal returns
`128 + signal`, including 130 for Ctrl-C. Metagross passes Ctrl-C and SIGTERM
on to the script and keeps tracing until the script exits, so a script that
handles the signal shuts down as it would untraced and the capture still ends
with its summary. If Metagross itself is killed, the kernel sends the script
SIGTERM, so the script does not run on unattended. Metagross returns 1 for validation,
dependency, privilege, probe, compile, attach, transport, or cleanup
failures, and 2 for invalid command-line syntax. A broken trace output stops
rendering but lets the target finish and preserves its status.
With `--web`, Metagross keeps serving the finished capture after the target
exits and returns the target's status once Ctrl-C or SIGTERM stops the
dashboard, so a non-interactive run blocks until it is signalled. A dashboard
that cannot start returns 1 before the target is forked.
For direct dashboard delivery, a failed startup handshake returns 1 without
executing the target. A delivery failure after the startup barrier has released
only warns and marks the in-memory capture incomplete; the target continues and
its status remains authoritative.


## Built-in web dashboard

`--web` streams the capture to a browser dashboard that Metagross starts
itself, with no trace file and no token to handle. It is a tracer option;
`view --web` is the separate viewer for saved JSONL files.

```sh
sudo /usr/bin/python3 -m metagross --web examples/gpu_demo.py
```

Before loading probes or forking the target, the controller starts the
in-memory receiver as the same unprivileged account the target runs as, in its
own session, bound to `127.0.0.1` on `--web-port` (8765 by default). It prints
the private viewer URL to standard error. The dashboard has no option to
listen on another address. To watch it from another machine, forward the same
port over SSH and open the printed URL there:

```sh
ssh -L 8765:127.0.0.1:8765 user@gpu-machine
```

The local port must equal the dashboard port, because the server rejects
requests addressed to any other port. The controller generates the producer
token and passes it to the receiver through an inherited pipe; the token never
appears in an environment variable, the command line, or the terminal. Terminal
Ctrl-C reaches the target, not the dashboard, and the dashboard exits if the
controller dies.

After the target exits, the dashboard keeps serving until Ctrl-C or SIGTERM,
then Metagross returns the target's exit status. SIGTERM before the target
exits, for example `docker stop` during a run, is passed to the script; when
the script exits, Metagross finishes the capture, stops the dashboard instead
of serving on, and returns the script's status. Add `--output` or
`--summary-output` when a durable copy is also required. In Docker, run the
container with `--network host` so `127.0.0.1` is the host's loopback; see the
[Docker guide](../examples/docker/README.md).

## Advanced direct delivery

Direct delivery is fileless and ephemeral. In the host terminal, generate one
secret and start the unprivileged, loopback-only receiver:

```sh
export METAGROSS_DASHBOARD_TOKEN="$(/usr/bin/python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
printf 'Copy this token into the Docker terminal: %s\n' "$METAGROSS_DASHBOARD_TOKEN"
/usr/bin/python3 -B -m metagross view --web --receive --port 8765
```

After exporting the copied token in the Docker terminal, run the already-built
example image:

```sh
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

`--network host` makes the container's numeric `127.0.0.1` reach the host
receiver and removes Docker's network-namespace isolation. It is separate from
the still-required `--pid=host`, which keeps eBPF process identity consistent.
Use both only with trusted local containers. The matching token is removed from
the controller environment before the target starts and is never sent to the
browser.

No `/traces` mount, JSONL file, or summary file is needed. The dashboard starts
at `WAITING`, changes to `LIVE`, and retains the completed capture in memory.
Stopping the host dashboard loses that state; a new authorized run replaces it
instead of merging counts. Add the existing `--output` and `--summary-output`
options when a durable recording is required. An unavailable or unauthorized
dashboard fails before target execution; a receiver lost after startup produces
one controller warning, marks delivery incomplete, and does not replace the
target's exit status.


The shell that exported the producer token retains its own copy. After both
commands finish, run `unset METAGROSS_DASHBOARD_TOKEN` in each such shell.
Generate a fresh token for a later session; do not put tokens in shell history,
shared logs, screenshots, source files, or capture files.

Direct delivery has its own completeness accounting. The HTTP-only final
summary adds `capture.delivery_dropped` and sets `complete` false if delivery
lost events. The receiver expects `capture.events - capture.delivery_dropped`
records. Local JSONL, summary files, and `--stats` retain capture completeness
without this transport-specific field. If the final summary cannot reach the
receiver, it cannot certify completion; inspect the controller warning too.

## Viewer option reference

| Option | Meaning and accepted values |
|--------|-----------------------------|
| `--snapshot` | Read a completed trace, print to stdout, and exit. |
| `--follow` | Follow JSONL in an interactive terminal. |
| `--web` | Serve a browser dashboard, loopback-only unless `--host` is given. |
| `--receive` | Web-only in-memory receiver; rejects a trace path or `--summary`. |
| `--summary FILE` | Optional version-1 final capture summary for file modes. |
| `--recent N` | Retain 1 to 10,000 recent events; default 500. |
| `--width COLUMNS` | Snapshot-only width, 60 to 240; default uses terminal width (120 fallback). |
| `--refresh SECONDS` | Follow/web-only interval, 0.05 to 5.0; default 0.2. |
| `--port PORT` | Web-only port, 0 to 65,535; default 8765, zero selects a free port. |
| `--host ADDRESS` | Web-only internal IPv4 bind address; default 127.0.0.1. Use 0.0.0.0 or an internal interface address to admit viewers on the internal network; public addresses are refused. With `--receive`, only 127.0.0.1 or 0.0.0.0. |
| `-h`, `--help` | Show viewer usage. |

Exactly one of `--snapshot`, `--follow`, or `--web` is required. A trace path is
required except with `--receive`. Invalid combinations return 2. Viewer I/O or
server startup failures return 1. A normal viewer exit returns 0; Ctrl-C in the
web server returns 130.

## Viewer status reference

| State | Meaning |
|-------|---------|
| `WAITING` | Trace file or first direct capture has not arrived. |
| `LIVE` | Events are available without a final matching summary. |
| `PAUSED` | Terminal follower consumption is paused. |
| `EVENTS ONLY` | Snapshot has valid events but no completeness summary. |
| `COMPLETE` | Summary declares completeness and event count matches. |
| `INCOMPLETE` | Capture or direct delivery reports loss/failure. Every viewer prints the reason, for example `7 profile records lost`. |
| `MISMATCH` | Observed event count disagrees with the summary. |
| `MALFORMED` or ` / MALFORMED` | Invalid lines were skipped; inspect the count. |
| `LIVE / SUMMARY ERROR` | Non-empty summary is invalid; retry occurs on change. |
| `ERROR` | Trace read or direct capture error prevents normal following. |

Without a summary, `LIVE` does not prove that a saved capture is still running.
An empty summary is pending. Live readers hold a partial final JSONL line until
its newline arrives and reset on trace replacement or truncation.

## Local HTTP interface

The browser state payload uses `schema_version: 1`. Its fields are
`generation`, `trace_name`, `status`, `waiting`, `trace_error`, `summary_error`,
`refresh_ms`, `incomplete_reasons`, `metrics`, `timeline`, `top_apis`,
`top_functions`, `top_kernels`, `recent_events`, and `memory_samples`.
`incomplete_reasons` is a list of sentences saying why the final summary
reports the capture incomplete; it is empty otherwise. It is a bounded display model, not a
replacement for the durable JSONL capture. The timeline represents retained
recent events, while aggregates summarize all successfully read events.

| Route | Methods | Credential and purpose |
|-------|---------|------------------------|
| `/`, static assets | GET, HEAD | Public application shell, with no trace data or credentials. |
| `/api/state` | GET, HEAD | Exactly one viewer bearer credential; current display state. |
| `/api/capture/start` | POST | Producer bearer credential; begin/replace one in-memory capture. |
| `/api/capture/events` | POST | Producer bearer credential; ordered bounded event batch. |
| `/api/capture/finish` | POST | Producer bearer credential; final summary. |
| `/api/capture/abort` | POST | Producer bearer credential; incomplete terminal state. |

Missing, wrong, or duplicate authorization headers return 401. Producer routes
are available only in receive mode. Tokens cannot substitute for each other's
role. Authentication uses a bearer header, never cookies or query parameters;
there is no cross-origin sharing policy. The private URL fragment is cleared
before the first state request. After a server restart, use its new private URL.

## Storage and display bounds

Viewer line reads are limited to 1 MiB and summary reads to 4 MiB. Aggregate
storage is bounded to 512 APIs, 4,096 functions, 4,096 kernels, and 4,096
spans, with overflow folded into other groups. Memory history retains 2,000 samples. The configured
recent-event bound is at most 10,000; the web payload can show a smaller subset.
The browser timeline and Events view cap that retained window at 1,000 events,
memory samples at 120, and each top-groups section at eight rows.
Display text and detail fields are capped and control characters are sanitized.
These bounds keep inspection usable but do not make hostile input harmless.

The tracer bounds what it keeps from the target's profile pipe: names longer
than 500 bytes are treated as a corrupt stream, at most 65,536 distinct project
frames are kept, and unread profile data is capped at 16 MiB, after which the
target drops and counts records. Calls to frames beyond that count are counted
as lost profile records and treated like any other loss: the frames known at
that point are forgotten and the capture is incomplete. On every loop
tick, whether or not the GPU is active, frame history older than 100 ms before
the last ring buffer drain and the oldest call still waiting is discarded,
nothing is kept for a thread whose history is empty, and at most 512 KiB of
profile data is decoded so the ring buffer keeps being read. A call waits at
most five seconds for its profile history, and at most 200,000 calls wait at
once; beyond either limit the oldest are written as `<unknown>`. If the tracer
stops reading
altogether, the script waits at most one second for it, then carries on and
drops profile records until the tracer reads again; the capture is marked
incomplete.

The dashboard server closes a connection that is silent for 10 seconds and
serves at most 32 connections at once; further clients are disconnected. This
bounds its threads and memory; it does not stop a local user from deliberately
occupying every slot.

The producer queue is bounded and its event offers do not block the trace loop.
Delivery accepts at most 128 events per batch, 1 MiB per batch, and 64 KiB per
event. Oversized or undeliverable events count as delivery drops. The producer
uses numeric `127.0.0.1` without DNS, redirects, or HTTP proxy discovery.

## Privilege and trust boundary

Run only trusted scripts and trusted local containers. The privileged controller
loads BPF programs and attaches probes; the target runs as the validated sudo
caller, or as root only with `--allow-root-target`. The child drops credentials
and sets `no_new_privs` before entering a fresh interpreter with the same PID,
so the target and its descendants cannot gain privileges through setuid
programs: a script that runs `sudo` works bare but fails under Metagross. Its profiling descriptor
is restored to close-on-exec before target code runs; controller descriptors do
not intentionally cross into target code. Arguments remain unchanged. The
bootstrap removes the producer token before the target can inspect its environment.

`python -m` puts the working directory first on Python's import path, and the
interpreter imports some of its own modules from there (`types`, and `runpy` on
Python 3.10) before any Metagross code runs. Metagross removes a working
directory that is not its own checkout from the path before importing anything
itself, but it cannot undo what the interpreter already loaded. Run
`sudo /usr/bin/python3 -m metagross` only from the checkout. To start Metagross
from any other directory, launch it by path, which never puts the working
directory on the import path:

```sh
sudo /usr/bin/python3 /path/to/metagross-checkout/metagross [options] script.py
```

This is a local diagnostic, not a sandbox, multi-user service, production
monitor, or isolation boundary. A malicious target, compromised invoking account,
root process, or compromised browser is outside its protection. Local trace data
can include paths, function/kernel names, pointer values, and timing information.
Protect capture directories and the private viewer URL; do not publish real
workload traces without reviewing and sanitizing their contents.
