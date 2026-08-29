# Development guide

Metagross is becoming a product, but the implementation should stay small,
predictable, and diagnosable. Favor a boring core over clever tracing magic.

## Architecture boundaries

- `metagross/__main__.py` is only the module entry point. Do not put product or
  tracing logic there.
- `metagross/__init__.py` owns CLI parsing, validation, privilege handling,
  forking, startup synchronization, BCC orchestration, output setup, and exit
  behavior.
- `metagross/_bpf.py` owns CUDA API definitions, libcuda discovery, symbol
  resolution, eBPF C generation, BCC loading, probe attachment, and raw structs.
- `metagross/_profile.py` owns the child-process Python profiler and binary
  profile record codec.
- `metagross/_events.py` owns stream joining, attribution, enrichment, allocation
  and kernel registries, and output rendering.
- Keep examples lightweight. They should demonstrate behavior, not become test
  harnesses or product code.

## CLI contract

- Metagross options must appear before the target script.
- Everything after the target script belongs to the target and must pass through
  unchanged.
- `--project-root` defaults to the current directory.
- Target scripts must resolve to regular `.py` files inside the project root.
- `--ebpf` must work without root, BCC, CUDA, libcuda discovery, or an NVIDIA
  driver.
- Do not write Metagross diagnostics to stdout during normal tracing; stdout is
  target-owned. Trace output defaults to stderr unless `--output` is provided.

## Privilege and safety

- Live tracing is privileged. Treat changes here as security-sensitive.
- Validate sudo metadata as a unit: `SUDO_UID`, `SUDO_GID`, and `SUDO_USER` must
  be complete, numeric where expected, and match passwd data.
- Drop the target to the invoking sudo user when valid metadata is available.
- Preserve `HOME`, `USER`, and `LOGNAME` for the dropped user.
- Keep inherited profile pipe file descriptors close-on-exec.
- Maintain the startup barrier: the child must not execute target code until the
  parent has attached every required uprobe/uretprobe to the exact child PID.
- Flush target stdout/stderr before child `os._exit`.
- Broken trace pipes must not kill the target.

## Output files

- New output files must be created with mode `0600`.
- New output files should be owned by the invoking user, not root, when running
  under sudo.
- Existing output paths may only be truncated when they are regular,
  non-symlink files owned by the invoking uid.
- Reject directories, symlinks, device files, FIFOs, and files owned by another
  user.

## eBPF and CUDA API changes

When adding or changing a traced CUDA API, update all relevant pieces together:

1. `_bpf.APIS` and `API_BY_ID` expectations.
2. eBPF argument capture and raw event layout.
3. Python `ctypes` structures that mirror eBPF structs.
4. `_events.describe` enrichment.
5. Table and JSONL rendering tests.
6. README traced API and detail tables.

Rules:

- `cuLaunchKernel` remains mandatory; other APIs should be optional unless there
  is a strong compatibility reason.
- Resolve CUDA symbol suffixes carefully and deduplicate addresses.
- Keep eBPF programs simple. Capture arguments in kernel space; interpret in
  Python.
- Never assume a kernel/function handle has a name. Kernel names are best-effort.

## Attribution rules

- Use native thread IDs and monotonic nanoseconds for joins.
- Do not use wall time for ordering profile and GPU events.
- Keep the GPU-event hold window unless replacing it with an equally conservative
  ordering strategy.
- Project-file detection must exclude Metagross itself, site packages,
  dist-packages, virtual-environment packages, and files outside the project
  root.
- Unknown or ambiguous attribution should render as unknown, not guessed.
- Mismatched profile call/return records should degrade conservatively and keep
  rendering.

## Productization guidance

- Before adding packaging, configuration, or service integrations, keep the CLI
  behavior and README examples as the source of truth.
- Avoid hidden global state that would make repeated invocations in tests or
  wrappers flaky.
- Prefer additive flags and backwards-compatible output fields.
- Keep product telemetry, upload, or sharing behavior opt-in if it is added later.
- Do not introduce network calls into the tracing path without explicit product
  requirements and tests.
