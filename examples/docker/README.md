# Dockerized PyTorch examples

This example builds Metagross, Ubuntu's BCC Python bindings, and a pinned CUDA
PyTorch wheel into one image. It is intended for local development and demos on
a CUDA-capable Linux host.

## Host requirements

- A native x86-64 Linux host with an NVIDIA GPU and a driver compatible with
  CUDA 12.4. RTX 50-series and other Blackwell GPUs need the CUDA 12.8 build
  described in [Build](#build).
- Docker with the NVIDIA Container Toolkit (`docker run --gpus all ...`).
- BPF enabled in the host kernel.
- Headers for the running host kernel at `/lib/modules/$(uname -r)/build`.
- Host debugfs/tracefs mounts available under `/sys/kernel`.

Confirm GPU passthrough before building:

```sh
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

## Build

From the repository root:

```sh
docker build \
  --build-arg TARGET_UID="$(id -u)" \
  --build-arg TARGET_GID="$(id -g)" \
  -f examples/docker/Dockerfile \
  -t metagross-pytorch .
```

The UID/GID arguments default to `1000`. The image's Metagross controller runs
as root for BPF access, while the target script is dropped to this unprivileged
account. Use a non-root UID/GID that is not already assigned in the base image.

The default image pins PyTorch 2.5.1 with CUDA 12.4 wheels. Override
`TORCH_VERSION` and `TORCH_INDEX_URL` together if another supported wheel is
needed.

Blackwell GPUs (compute capability 12.0, such as the RTX 5090) require this
override, because the CUDA 12.4 wheels contain no kernels for them. This build
was verified on an RTX 5090 with driver 580:

```sh
docker build \
  --build-arg TARGET_UID="$(id -u)" \
  --build-arg TARGET_GID="$(id -g)" \
  --build-arg TORCH_VERSION=2.11.0 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  -f examples/docker/Dockerfile \
  -t metagross-pytorch .
```

The build resumes interrupted wheel downloads. If pulling the `nvidia/cuda`
base image itself fails partway, pull it on another machine and transfer it.
The `--platform` option needs Docker 28 or newer:

```sh
docker save --platform linux/amd64 -o cuda-base.tar nvidia/cuda:12.4.1-runtime-ubuntu22.04
docker load -i cuda-base.tar
```

## Available workloads

All workloads use seeded synthetic inputs and core PyTorch; no datasets, model
downloads, torchvision, or network access are required at run time. Exact
floating-point values can still vary across GPU models, drivers, and PyTorch
kernels.

| Script | What it demonstrates |
|---|---|
| `basic_tensor_ops.py` | Distinct upload, matrix compute, synchronization, and download phases. |
| `training_step.py` | A small MLP training loop and expected autograd attribution limits. |
| `basic_cnn.py` | Convolution, pooling, classifier training, and an evaluation download. |
| `basic_vit.py` | Patch embedding, multi-head attention, transformer training, and classification. |
| `basic_decoder.py` | Decoder-only causal attention, prompt prefill, and token-by-token decoding. |
| `complex_pipeline.py` | CPU preparation, pinned memory, asynchronous upload, a CNN/transformer hybrid, gradient accumulation, optimizer work, and evaluation. |

The examples are intentionally small enough for a local smoke test. They are not
performance benchmarks, accuracy benchmarks, or reference model
implementations.

## Run the basic workload

```sh
docker run --rm \
  --gpus all \
  --privileged \
  --pid=host \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v /sys/kernel/tracing:/sys/kernel/tracing \
  metagross-pytorch
```

The target prints its checksum to stdout. Metagross prints the trace table to
stderr.

The elevated flags are important:

- `--privileged` grants the broad BPF/perf permissions needed across supported
  Docker and kernel combinations. Use this example only with trusted local code.
- `--pid=host` keeps the child PID used by Metagross consistent with the TGID
  observed by eBPF's exact-process filter.
- The `/lib/modules` and `/usr/src` read-only mounts expose the running host
  kernel's modules and headers to BCC's compiler.
- The `/sys/kernel/debug` and `/sys/kernel/tracing` mounts let BCC register and
  clean up uprobe events. Omitting them can produce a successful trace followed
  by a cleanup failure.

## Stream directly to the host browser dashboard

This path sends normalized events directly to a loopback-only dashboard without
creating a JSONL or summary file. Rebuild the image after source changes, then
generate one token and start the receiver in the host terminal:

```sh
export METAGROSS_DASHBOARD_TOKEN="$(/usr/bin/python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
printf 'Copy this token into the Docker terminal: %s\n' "$METAGROSS_DASHBOARD_TOKEN"
/usr/bin/python3 -B -m metagross view --web --receive --port 8765
```

Export the copied value as `METAGROSS_DASHBOARD_TOKEN` in the Docker terminal,
then run:

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

Do not mount `/traces` and do not pass `--output` or `--summary-output` for the
fileless workflow. Open the printed host URL: it moves from `WAITING` to `LIVE`
and then to a final status while the target checksum remains on container
stdout.

`--network host` is required because the privileged producer connects only to
numeric `127.0.0.1`; it removes Docker network isolation. `--pid=host` remains
separately required so the PID filtered by Metagross matches the TGID observed
by host eBPF. Use these privileges and host namespaces only with trusted local
code. The shared token is removed before the target script runs and is not
available to browser JavaScript.

Receiver state exists only in memory and is lost when the host dashboard stops.
A new authorized container run replaces the retained capture rather than
merging with it. Use the existing `--output` and `--summary-output` options, plus
a host volume, when durable JSONL and summary files are required. Missing,
wrong-token, or unreachable receivers fail the handshake before target
execution. Losing the receiver after startup emits one controller warning,
marks delivery incomplete, and preserves target stdout, stderr, and exit status.

## Run another workload

Select any script from the table above:

```sh
WORKLOAD=basic_vit.py
docker run --rm \
  --gpus all \
  --privileged \
  --pid=host \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v /sys/kernel/tracing:/sys/kernel/tracing \
  metagross-pytorch \
  --project-root /workspace/workloads \
  "/workspace/workloads/$WORKLOAD"
```

The training examples deliberately synchronize at stage boundaries so waits are
easy to identify. Some backward-pass launches may show unknown attribution
because PyTorch autograd can launch them from C++ threads without an active
Python project frame. The decoder is deliberately implemented without a KV
cache so repeated token-by-token work remains visible as an understandable
launch pattern.

## Run the workload correctness tests

The image includes GPU tests for tensor values, model output shapes, finite
losses, parameter updates, decoder causality, generated-token bounds, pinned
memory, transfer integrity, and pipeline evaluation:

```sh
docker run --rm --gpus all \
  --user metagross-target \
  --entrypoint /usr/bin/python3 \
  metagross-pytorch \
  -m unittest discover -s /workspace/tests -v
```

These tests need GPU passthrough but not root, BPF privileges, host PID access,
or kernel filesystem mounts. They verify workload structure and numerical
invariants independently of the Metagross tracer; they do not establish model
quality or benchmark accuracy.

## Write JSONL to the host

Use the image built with your host UID/GID in [Build](#build). The command below
creates a private directory regardless of umask. For an existing `traces`
directory, `mkdir` leaves permissions unchanged; run `chmod 700 traces` if you
own it and intend to keep it private, or use a new directory and update the mount.

```sh
mkdir -p -m 700 traces
docker run --rm \
  --gpus all \
  --privileged \
  --pid=host \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v /sys/kernel/tracing:/sys/kernel/tracing \
  -v "$PWD/traces:/traces" \
  metagross-pytorch \
  --json --output /traces/basic.jsonl \
  --stats --summary-output /traces/basic-summary.json \
  --project-root /workspace/workloads \
  /workspace/workloads/basic_tensor_ops.py
```

Both files are created with mode `0600` and ownership matching the target
UID/GID selected at image build time. The summary is a versioned JSON document;
the event file retains the stable JSONL event schema.

Render an unprivileged terminal snapshot from the host checkout:

```sh
python3 -m metagross view \
  --snapshot --summary traces/basic-summary.json \
  traces/basic.jsonl
```

Or start the live dashboard before running the Docker capture in another
terminal. It waits if the event file does not exist yet:

```sh
python3 -m metagross view \
  --follow --summary traces/basic-summary.json \
  traces/basic.jsonl
```

For the responsive browser dashboard instead, run:

```sh
python3 -m metagross view \
  --web --summary traces/basic-summary.json \
  traces/basic.jsonl
```

Open the printed loopback URL in a browser. Use `--port PORT` when port 8765 is
already occupied.

Press `p` to pause file consumption and `q` to quit in the terminal dashboard.
Follow mode needs an interactive terminal of at least 80 columns by 18 rows.
The browser dashboard needs no TTY and provides a function-lane CUDA API
timeline with search, family filters, zoom/pan, event selection, source details,
Pause, and Refresh controls. Its timeline represents CPU-side API call timing,
not GPU kernel execution. Both viewers only read the saved files; they do not
need Docker, root, BCC, CUDA, or
access to the GPU. The `traces/` directory is ignored by Git because traces can
contain sensitive source paths and timing data.

## Run a workload without tracing

GPU passthrough itself does not require the BPF privileges:

```sh
docker run --rm --gpus all \
  --user metagross-target \
  --entrypoint /usr/bin/python3 \
  metagross-pytorch /workspace/workloads/basic_tensor_ops.py
```

## Benchmark tracing overhead

The host-side benchmark runner alternates bare and traced Docker runs and writes
machine-readable samples, medians, event-loss counters, and host/image metadata.
Rebuild the image after changing Metagross, then run a focused comparison from
the repository root:

```sh
python3 examples/docker/benchmark_overhead.py \
  --image metagross-pytorch \
  --workload basic_tensor_ops.py \
  --workload training_step.py \
  --repetitions 3 \
  --warmups 1 \
  --output .benchmarks/overhead.json
```

The four default modes are:

- `bare`: the workload under `/usr/bin/python3`, without Metagross.
- `full`: all API families with Python attribution.
- `no-attribution`: all API families without the Python profiling hook.
- `launch-only`: launch and kernel-name probes without Python attribution.

Use repeated `--mode` options to select fewer modes. With no `--workload`
options, the runner benchmarks all six portfolio workloads, which can take
several minutes.

Three steady-state probes isolate Python calls, rapid launches, and sustained
GPU compute. Each reports both end-to-end container time and an internal target
median that excludes initialization:

```sh
python3 examples/docker/benchmark_overhead.py \
  --image metagross-pytorch \
  --workload python-calls \
  --workload rapid-launches \
  --workload compute-heavy \
  --repetitions 3 \
  --warmups 0 \
  --output .benchmarks/steady-state.json
```

The comparison is hardware- and software-specific; it is an overhead regression
tool, not a model-performance benchmark. A nonzero lost-event count means that
capture was incomplete.

## Troubleshooting

- `CUDA is unavailable`: verify the NVIDIA Container Toolkit and the
  `--gpus all` flag.
- `no kernel image is available for execution on the device`, or a warning that
  the GPU's CUDA capability is not compatible with the installed PyTorch: build
  with a PyTorch wheel that supports the GPU; see the Blackwell example in
  [Build](#build).
- Missing kernel headers or BCC compile errors: install headers for `uname -r`
  on the host and verify `/lib/modules/$(uname -r)/build` exists.
- `open(/sys/kernel/debug/tracing/uprobe_events)` or cleanup failures: include
  both `/sys/kernel/debug` and `/sys/kernel/tracing` mounts.
- No attributed events or immediate attach failure: verify `--pid=host` was
  included.
- `libcuda.so.1 not found`: verify the container is launched with NVIDIA GPU
  passthrough; the NVIDIA runtime supplies the host driver library.
- Direct delivery returns HTTP 401 or fails before the workload starts: export
  the same `METAGROSS_DASHBOARD_TOKEN` in both terminals and start the receiver
  before the container.
- Direct delivery cannot connect: include `--network host`, keep the receiver
  port and `--dashboard-port` equal, and verify the host port is unused.

Docker Desktop on macOS and Windows does not expose a native NVIDIA Linux driver
and host eBPF environment suitable for this example.
