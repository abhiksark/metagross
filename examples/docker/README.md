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

The base image, PyTorch, NumPy, and pip are pinned by version, not by image
digest or package hash, so the transfer above keeps working. Pin them yourself
if you need a reproducible supply chain.

## Use the metagross command

[`metagross`](metagross) is a short shell script that runs the image on a script
in the current directory. Install it once from the repository root:

```sh
mkdir -p ~/.local/bin
ln -s "$PWD/examples/docker/metagross" ~/.local/bin/metagross
```

Then, from any project directory:

```sh
metagross run.py            # trace table on stderr
metagross --web run.py      # live browser dashboard
metagross                   # the included basic workload
```

A call that traces a script runs this command, with every argument passed to
Metagross unchanged:

```sh
docker run --rm -i --gpus all --privileged --pid=host \
  -v /lib/modules:/lib/modules:ro -v /usr/src:/usr/src:ro \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v /sys/kernel/tracing:/sys/kernel/tracing \
  -v "$PWD:$PWD" -w "$PWD" $METAGROSS_DOCKER_ARGS \
  metagross-pytorch "$@"
```

- The current directory is mounted read-write at the same path and becomes the
  working directory. The default `--project-root .` is therefore your project,
  relative paths resolve as they do on the host, and files your script writes
  there belong to the build-time UID and GID. Other host directories are not
  mounted, so paths outside the current directory do not resolve. That is not
  isolation: the container is privileged, shares the host PID namespace, and
  mounts host kernel directories, so running it is equivalent to root on the
  host.
- `--web` before the script adds `--network host` so your browser can reach the
  dashboard. After the script it is one of the script's own arguments.
- `METAGROSS_IMAGE` selects another image tag, for example
  `METAGROSS_IMAGE=metagross-pytorch:cu128 metagross run.py`.
- `METAGROSS_DOCKER_ARGS` adds `docker run` options. Your shell's environment
  variables and other directories do not reach the script unless you pass
  them, for example
  `METAGROSS_DOCKER_ARGS="-e CUDA_VISIBLE_DEVICES=1 -v /data:/data:ro" metagross run.py`.
- `metagross view ...` runs the viewers as you, with only the current
  directory mounted and no privileges, GPU, or kernel mounts, so the trace
  files must be under the current directory. `--help`, `--version`, and
  `--ebpf` before the script also run without privileges.
- The command refuses to run from `/workspace` or a directory below it, where
  the image keeps its own files.
- `--output` and `--summary-output` refuse a path if your group or other
  users can write to the current directory or to any directory between it and
  the output file. With a umask of 002, the default on Ubuntu, the current
  directory is itself group-writable, so a private subdirectory inside it does
  not help. Run `chmod g-w .` in the project directory first, or use the full
  `docker run` form in [Write JSONL to the host](#write-jsonl-to-the-host),
  which mounts a private `traces` directory by itself.

The sections below use the full `docker run` form and explain each flag.

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

## Watch the capture in the browser dashboard

Add `--network host` and `--web`. Metagross starts the dashboard inside the
container as the unprivileged target account and prints a private URL:

```sh
docker run --rm \
  --gpus all \
  --privileged \
  --pid=host \
  --network host \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v /sys/kernel/tracing:/sys/kernel/tracing \
  metagross-pytorch \
  --web \
  --project-root /workspace/workloads \
  /workspace/workloads/basic_tensor_ops.py
```

Open the printed `http://127.0.0.1:8765/#viewer_token=…` URL on the host. It
moves from `WAITING` to `LIVE` and then to a final status while the target
checksum remains on container stdout. After the workload exits, the dashboard
keeps the capture until you press Ctrl-C or stop the container; the container
then exits with the workload's status. Use `--web-port PORT` when port 8765 is
taken.

`--network host` makes the dashboard's `127.0.0.1` the host's loopback and
removes Docker's network isolation. `--pid=host` remains separately required so
the PID filtered by Metagross matches the TGID observed by host eBPF. Use these
privileges and host namespaces only with trusted local code. The producer token
is generated inside the container and is never printed, placed in the
environment, or sent to the browser.

The capture is kept in memory only. Add `--output` and `--summary-output` with a
host volume, as in [Write JSONL to the host](#write-jsonl-to-the-host), when
durable files are also required. A dashboard that cannot start, for example
because its port is taken, fails before the workload runs. For the lower-level
two-terminal receiver, see
[Advanced direct delivery](../../docs/reference.md#advanced-direct-delivery).

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
machine-readable samples, medians, loss counters for events and for profile
records, and host/image metadata.
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
tool, not a model-performance benchmark. A nonzero count in any of the three
loss columns means that capture was incomplete.

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
- `not in the host PID namespace`, or an immediate attach failure: add
  `--pid=host`.
- `libcuda.so.1 not found`: verify the container is launched with NVIDIA GPU
  passthrough; the NVIDIA runtime supplies the host driver library.
- `cannot start the web dashboard`: another process holds the port; pass
  `--web-port` with a free port.
- The dashboard URL does not load: include `--network host` and open the URL on
  the machine running the container. From another machine, forward the port
  first: `ssh -L 8765:127.0.0.1:8765 user@gpu-machine`.

Docker Desktop on macOS and Windows does not expose a native NVIDIA Linux driver
and host eBPF environment suitable for this example.
