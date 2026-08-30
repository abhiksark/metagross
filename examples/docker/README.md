# Dockerized PyTorch examples

This example builds Metagross, Ubuntu's BCC Python bindings, and a pinned CUDA
PyTorch wheel into one image. It is intended for local development and demos on
a CUDA-capable Linux host.

## Host requirements

- A native x86-64 Linux host with an NVIDIA GPU and a driver compatible with
  CUDA 12.4.
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

```sh
mkdir -p traces
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

Render an unprivileged terminal dashboard from the host checkout:

```sh
python3 -m metagross view \
  --snapshot --summary traces/basic-summary.json \
  traces/basic.jsonl
```

The viewer only reads the saved files; it does not need Docker, root, BCC, CUDA,
or access to the GPU after capture.

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
- Missing kernel headers or BCC compile errors: install headers for `uname -r`
  on the host and verify `/lib/modules/$(uname -r)/build` exists.
- `open(/sys/kernel/debug/tracing/uprobe_events)` or cleanup failures: include
  both `/sys/kernel/debug` and `/sys/kernel/tracing` mounts.
- No attributed events or immediate attach failure: verify `--pid=host` was
  included.
- `libcuda.so.1 not found`: verify the container is launched with NVIDIA GPU
  passthrough; the NVIDIA runtime supplies the host driver library.

Docker Desktop on macOS and Windows does not expose a native NVIDIA Linux driver
and host eBPF environment suitable for this example.
