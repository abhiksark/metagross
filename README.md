<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
  <img src="assets/logo-light.svg" alt="" width="240">
</picture>

[![Tests](https://github.com/abhiksark/metagross/actions/workflows/test.yml/badge.svg)](https://github.com/abhiksark/metagross/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![eBPF](https://img.shields.io/badge/eBPF-BCC-orange)
![CUDA](https://img.shields.io/badge/NVIDIA-CUDA%20driver%20API-76b900)
![Status](https://img.shields.io/badge/status-experimental-yellow)

# Metagross

Watch selected CUDA driver calls update live, grouped by the Python project
function that caused them. Metagross reports host-side API elapsed time, not
GPU kernel execution time or utilization.

This experimental, source-only public preview traces one trusted Python workload
without changing normal target stdout; trace rows go to stderr by default.

<img src="assets/dashboard-demo.webp" alt="Metagross web dashboard filling with CUDA driver calls from a PyTorch pipeline, grouped by Python function, then showing one cuBLAS launch's details and the summary panels" width="1000">

*Recorded trace of a PyTorch pipeline (537 CUDA driver calls) replayed through
the web dashboard at about 1.3× speed. Calls without a safe project frame stay
`<unknown>`.*

## Functionality overview

| Capability | What it does |
|------------|--------------|
| [Driver call tracing](docs/reference.md#traced-api-table) | Hooks 17 CUDA driver APIs with eBPF uprobes: kernel launches, allocations and frees, copies between host and device, and stream, context, and event synchronization. Needs no changes to the traced script; `--trace` selects any of the `launch`, `memory`, `copy`, and `sync` families. |
| [Python attribution](docs/reference.md#how-attribution-works) | Assigns each call to the project function active at API entry, skipping standard-library and installed-package frames. Calls with no safe project frame stay `<unknown>`; `--no-attribution` turns the profile hook off. |
| [Call details](docs/reference.md#traced-api-table) | Records the return code, CPU-side duration, and per-API arguments: launch grid, block, shared memory, and stream; byte counts; pointers; and resolved kernel names. |
| [Named regions](docs/reference.md#op-spans) | Tags every call inside `with metagross.span("name"):` with that label, alongside function attribution. The span is a no-op when the script runs outside a trace. |
| [Live web dashboard](#what-the-dashboard-shows) | Shows the capture in a browser as it runs: a function-grouped timeline with zoom, pan, and filters, an event table with a detail inspector, and API, function, kernel, and allocation summaries. |
| [Terminal viewers](docs/reference.md#viewer-option-reference) | Prints a static dashboard with `view --snapshot` or follows a growing JSONL file with `view --follow`. Both run without root, BCC, CUDA, or a GPU. |
| [Durable output](docs/reference.md#output) | Writes table rows to stderr by default, stable JSONL with `--json --output`, a versioned summary with `--summary-output`, and a one-line capture report with `--stats`. |
| [Completeness checks](docs/reference.md#viewer-status-reference) | Counts lost events, dropped nested calls, and CUDA errors. Only a reconciled final summary is shown as `COMPLETE`. |
| [Target preservation](docs/reference.md#privilege-and-trust-boundary) | Runs one script as the invoking sudo user, with its arguments, stdout, and exit status unchanged. |

<a id="quick-start"></a>
## Quick start: trace your script live

Live tracing needs Linux, an NVIDIA GPU and compatible driver, root/sudo, BPF
support, running-kernel headers, and Python 3.10+. On Ubuntu, use the system
Python and BCC bindings:

```sh
git clone https://github.com/abhiksark/metagross.git
cd metagross
sudo apt install python3-bpfcc "linux-headers-$(uname -r)"
```

Run both terminals from this repository directory.

1. **Terminal one: start the unprivileged loopback receiver.**

   ```sh
   export METAGROSS_DASHBOARD_TOKEN="$(/usr/bin/python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
   printf 'Copy this token into the tracing terminal: %s\n' "$METAGROSS_DASHBOARD_TOKEN"
   /usr/bin/python3 -B -m metagross view --web --receive --port 8765
   ```

   Open the complete private URL printed by the receiver.
   Its `#viewer_token=…` browser credential differs from the producer token and
   disappears from the visible URL after entering browser session storage.
   A new tab needs the complete URL again, or paste it into the field the page
   shows.
   Keep the URL private. To view it from another machine on a trusted internal
   network, add `--host 0.0.0.0` instead of proxying the port, and read the
   warning it prints. Public addresses and public-internet clients are refused
   as defense in depth; this does not replace a firewall.

2. **Terminal two: trace your script** with the copied producer token.

   The controller loads BPF as root but drops the target to the validated sudo
   caller. Run only trusted local scripts; preserve only the producer variable
   across sudo:

   ```sh
   export METAGROSS_DASHBOARD_TOKEN='paste the copied token here'
   PROJECT_ROOT="/absolute/path/to/your/project"
   WORKLOAD="$PROJECT_ROOT/path/to/workload.py"
   sudo --preserve-env=METAGROSS_DASHBOARD_TOKEN \
     /usr/bin/python3 -B -m metagross \
     --dashboard-port 8765 \
     --project-root "$PROJECT_ROOT" \
     "$WORKLOAD"
   ```

Watch `WAITING → LIVE → COMPLETE` when the final summary reconciles. Target
stdout stays target-owned; trace rows remain on stderr by default. Keep the
receiver and browser tab open: rerunning the producer replaces the prior
in-memory capture and repopulates the same tab automatically.
`LIVE`, `INCOMPLETE`, `MISMATCH`, or malformed warnings are not final evidence
of a loss-free capture.

When finished, stop the receiver with Ctrl-C and clear the token in **both shells**:

```sh
unset METAGROSS_DASHBOARD_TOKEN
```

Direct delivery is in-memory: stopping the receiver loses the capture. For a
durable recording, see [JSONL and summary output](docs/reference.md#output).

## Point Metagross at your workload

- `PROJECT_ROOT` is the absolute project boundary used for attribution.
- `WORKLOAD` must be a regular `.py` file inside that root; `/usr/bin/python3`
  must be able to import its dependencies.
- Put Metagross options before `"$WORKLOAD"`; append target arguments after it
  and they pass through unchanged.

One launched Python process is traced; there is no existing-PID attach,
subprocess capture, or `python -m` target form. See the
[complete CLI and interpreter contract](docs/reference.md#usage) or the
[prepared-container route](examples/docker/README.md).

Optionally mark named regions of your own code with `import metagross` and
`with metagross.span("step"): ...`; every CUDA call attributed inside that
block carries `"step"` in its `span` field, in table output and JSONL alike.
See [op spans](docs/reference.md#op-spans) and the annotated
[`examples/quicklook.py`](examples/quicklook.py).

## What the dashboard shows

| Label | Meaning |
|-------|---------|
| Activity | Events / event rate, attributed percentage, and nonzero raw CUDA return codes. |
| Host timing | Accumulated CPU API elapsed time and its synchronization-call subset. |
| Data / memory | Successful copied bytes and current/peak successfully observed driver allocations, not device-wide utilization or tensor memory. |
| Views | Bounded recent events grouped by function, plus aggregate API, function, and kernel rankings. |

Only a reconciled final summary yields `COMPLETE`; `INCOMPLETE`, `MISMATCH`, or
malformed warnings mean the capture cannot be treated as complete.
`LIVE` is not final evidence of a loss-free capture.
See the [HTTP fields and schema](docs/reference.md#local-http-interface) and
[storage and display bounds](docs/reference.md#storage-and-display-bounds).

## Other ways to start

- [Sanitized no-GPU dashboard replay](examples/captures/README.md).
- [Prepared Docker workload, alternate workloads, and benchmarks](examples/docker/README.md).
- [Durable JSONL and final summary capture](docs/reference.md#output) and
  [snapshot/follow viewers](docs/reference.md#viewer-option-reference).

Both help commands run without root, BCC, CUDA, or a GPU:

```sh
/usr/bin/python3 -m metagross --help
/usr/bin/python3 -m metagross view --help
```

## When Metagross fits

| Question | Tool |
|----------|------|
| Which project function launched, copied, allocated, or synchronized through CUDA? | Metagross: connect selected driver calls to active Python project frames. |
| How do GPU execution and CPU/GPU overlap behave? | [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems). |
| Why is an individual GPU kernel slow? | [Nsight Compute](https://developer.nvidia.com/nsight-compute). |
| Which operators, autograd work, or tensor allocations dominate? | A framework profiler. |

## Requirements and limits

- Selected CUDA driver APIs in one launched Python process, including its Python
  threads; no framework-wide coverage or arbitrary-image entrypoint replacement.
- Timing is host-side API elapsed time, not GPU execution time or utilization.
- Attribution and kernel names are best effort; C++ worker threads without active
  project Python frames can be unknown.
- Observed driver allocations are not tensor memory; caching allocators reuse them,
  and VMM / expandable-segment allocations are outside the traced API set.
- Profiling adds overhead; high rates and nested driver re-entry can lose events.
  Inspect warnings and final summary completeness.
- Trusted local workloads only: this is not a sandbox or production monitor.
  Keep captures and the private viewer URL private.

See the [BCC installation guide](https://github.com/iovisor/bcc/blob/master/INSTALL.md),
[full limits](docs/reference.md#overhead-and-limits), and
[privilege and trust boundary](docs/reference.md#privilege-and-trust-boundary).

## Documentation

- [Reference](docs/reference.md): all options, schemas, API coverage, file safety,
  viewer states, exit behavior, and extended limits.
- [Examples](examples/README.md): local demonstrations and function descriptions.
- [Example captures](examples/captures/README.md): sanitized replay provenance.
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
