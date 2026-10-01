# Development guide

Metagross is becoming a product, but the implementation should stay small,
predictable, and diagnosable. Favor a boring core over clever tracing magic.

## Architecture boundaries

- `metagross/__main__.py` is only the module entry point. Do not put product or
  tracing logic there.
- `metagross/__init__.py` owns CLI parsing, validation, privilege handling,
  forking, startup synchronization, BCC orchestration, output setup, and exit
  behavior.
- `metagross/_target.py` is the private runner entered by exec after the attach
  barrier. It installs profiling and runs the target with normal interpreter
  finalization, including non-daemon thread joins and `atexit` handlers.
- `metagross/_dashboard.py` is the private runner for `--web`. The controller
  execs it as the target's account in a new session; it arms a parent-death
  signal, reads the producer token from an inherited pipe, and serves the
  in-memory receiver on `127.0.0.1`.
- `metagross/_bpf.py` owns CUDA API definitions, libcuda discovery, symbol
  resolution, eBPF C generation, BCC loading, probe attachment, and raw structs.
- `metagross/_profile.py` owns the child-process Python profiler and binary
  profile record codec.
- `metagross/_viewer.py` owns unprivileged trace-file parsing, bounded viewer
  state, summary reconciliation, terminal sanitization, and static rendering.
- `metagross/_follow.py` owns bounded incremental reads, partial JSONL records,
  truncation/replacement detection, and final-summary watching.
- `metagross/_tui.py` owns the standard-library curses dashboard, live overview
  layout, refresh loop, and keyboard controls.
- `metagross/_web.py` owns the standard-library HTTP server (loopback by
  default, reachable from the internal network only through the `--host`
  opt-in),
  bounded file and authenticated in-memory state, browser JSON payload, and
  embedded static frontend.
- `metagross/_publish.py` owns bounded ordered delivery from the privileged
  controller to the local dashboard.
- `metagross/_events.py` owns stream joining, attribution, enrichment, allocation
  and kernel registries, canonical event records, and output rendering.
- Keep examples lightweight. They should demonstrate behavior, not become test
  harnesses or product code.

## CLI contract

- Metagross options must appear before the target script.
- Everything after the target script belongs to the target and must pass through
  unchanged.
- Top-level `-h` and `--help` print usage without a target or privileged imports.
  Help flags after the target path remain target arguments.
- `--project-root` defaults to the current directory.
- Target scripts must resolve to regular `.py` files inside the project root.
- `--ebpf` must work without root, BCC, CUDA, libcuda discovery, or an NVIDIA
  driver.
- `view` must route before live-trace validation and work without root, BCC,
  CUDA, libcuda discovery, or an NVIDIA driver.
- Do not write Metagross diagnostics to stdout during normal tracing; stdout is
  target-owned. Trace output defaults to stderr unless `--output` is provided.

## Privilege and safety

- Live tracing is privileged. Treat changes here as security-sensitive.
- Validate sudo metadata as a unit: `SUDO_UID`, `SUDO_GID`, and `SUDO_USER` must
  be complete, numeric where expected, and match passwd data.
- Drop the target to the invoking sudo user when valid metadata is available.
- Preserve `HOME`, `USER`, and `LOGNAME` for the dropped user.
- The profile write descriptor crosses the single exec into the private runner.
  Restore close-on-exec there before target code runs; target exec descendants
  must not inherit it. Other controller descriptors remain close-on-exec.
- Maintain the startup barrier: the child must not execute target code until the
  parent has attached every required uprobe/uretprobe to the exact child PID and
  any explicitly requested dashboard has acknowledged capture start.
- Drop credentials before exec and preserve the child PID across that boundary.
  Set `no_new_privs` after the drop so the target cannot regain privileges.
  Use the controller's interpreter and locate the runner from this installation.
- Allow the target interpreter to finalize normally. Reserve child `os._exit`
  for startup failure before exec, flushing startup diagnostics first.
- Broken trace pipes must not kill the target.
- Pop `METAGROSS_DASHBOARD_TOKEN` before forking and defensively remove it again
  in the child before target code runs. Never put it in diagnostics, URLs,
  browser assets, API state, or response bodies.

## Output files

- New output files must be created with mode `0600`.
- New output files should be owned by the invoking user, not root, when running
  under sudo.
- Existing output paths may only be truncated when they are regular,
  non-symlink files owned by the invoking uid.
- Reject directories, symlinks, device files, FIFOs, and files owned by another
  user.

## Direct dashboard transport

- Generate an independent random viewer token for each dashboard server process,
  including file-backed mode. Print it only in the private URL fragment. Browser
  bootstrap must save it in session storage and clear the fragment before state
  requests; a new process invalidates old viewer credentials. The page may also
  accept a pasted token or private URL when the tab has no valid credential; it
  follows the same session-storage rule and must never write the token into the
  URL or history.
- Require exactly one matching bearer credential for GET and HEAD `/api/state`.
  Reuse constant-time credential comparison for viewer and producer requests,
  but never accept either token in the other's role. Keep static assets public
  and free of trace data and credentials. Do not add cookies, query-token auth,
  CORS, or producer credentials to browser resources.
- The viewer token protects trace data from callers without the secret. It is
  not isolation from root, the same compromised user, terminal readers, or
  browser/session-storage compromise. Bind loopback by default.
- `view --web --host` is the only internal-network opt-in. It must be a numeric
  IPv4 address, must print the plain-HTTP warning, and widens only viewer
  access. `--host` refuses public (`is_global`) addresses and loopback addresses
  other than `127.0.0.1`, and the viewer paths refuse clients whose address is
  public before any other Metagross check. These are defense in depth, not a
  substitute for a firewall.
  With a non-loopback bind, the Host check may also accept numeric IPv4 Host
  values but must keep rejecting DNS names, which is the DNS-rebinding defense.
- Capture ingest (`POST /api/capture/*`) stays loopback-only regardless of
  `--host`: reject non-loopback clients and keep the strict loopback Host check
  for ingest. `--receive` accepts only `--host 127.0.0.1` or `0.0.0.0`, because
  the producer connects only to `127.0.0.1`.
- `--dashboard-port` and `--web` are the only producer opt-ins. A token in the
  environment alone must not enable network activity.
- `--web` starts its receiver before BPF compilation and the target fork, with
  the target's credentials applied by `subprocess.Popen` (groups, gid, then
  uid), in a new session, with stdout on `/dev/null`. The controller generates
  the producer token and passes it only through an inherited pipe: never the
  environment, argv, or the terminal. The runner arms `PR_SET_PDEATHSIG` after
  the credential drop and exits if the controller is already gone. Stop the
  receiver on every controller exit path; after a traced run, keep serving
  until Ctrl-C or SIGTERM and return the target's status.
- The privileged producer may connect only to numeric IPv4 `127.0.0.1` at the
  validated port with `http.client`; do not add DNS, proxy, redirect,
  non-loopback, or browser-token paths.
- Keep JSON encoding and HTTP outside the trace loop. The trace loop may only
  normalize an event and perform a nonblocking offer to a bounded FIFO.
- Capture start is fail-closed while the target is behind the startup barrier.
  Delivery loss after release is fail-open for the target, bounded for the
  controller, declared in the HTTP-only summary, and must not change target
  stdout, stderr, signals, or exit status.
- The receiver binds `127.0.0.1` by default (or `0.0.0.0` with `--host`), keeps
  capture ingest loopback-only, requires the matching bearer token, validates
  complete bounded batches before one locked commit, and retains only one
  in-memory capture. A new authorized start replaces the prior capture.
- Docker direct delivery uses both `--network host` for loopback connectivity
  and the independently required `--pid=host` for eBPF identity. Host networking
  removes Docker network isolation, so this workflow is only for trusted local
  containers.
- Existing table, JSONL, summary-file, snapshot, follow, and file-backed web
  paths remain independent. Direct delivery creates no durable file unless an
  existing output option is explicitly supplied.

## eBPF and CUDA API changes

When adding or changing a traced CUDA API, update all relevant pieces together:

1. `_bpf.APIS` and `API_BY_ID` expectations.
2. eBPF argument capture and raw event layout.
3. Python `ctypes` structures that mirror eBPF structs.
4. `_events.describe` enrichment.
5. Table and JSONL rendering tests.
6. The traced API and detail tables in `docs/reference.md`.

Rules:

- `cuLaunchKernel` remains mandatory; other APIs should be optional unless there
  is a strong compatibility reason.
- Resolve CUDA symbol suffixes carefully and deduplicate addresses.
- Keep eBPF programs simple. Capture arguments in kernel space; interpret in
  Python.
- Never assume a kernel/function handle has a name. Kernel names are best-effort.

## Profile wire-format changes

The profile pipe (`metagross/_profile.py`) uses a versioned record format: a
6-byte common header (`rtype`, per-child `seq`, reserved byte) followed by a
per-`rtype` body. `RecordReader.feed` decodes it into tagged tuples that
`Joiner.on_profile_record` dispatches on. When adding or changing a profile
record type, update all relevant pieces together:

1. The `rtype` constant and its body `struct.Struct` in `_profile.py`.
2. The encoder (e.g. `encode_frame`, `encode_span`) that packs the
   common header plus the body.
3. `RecordReader.feed`'s dispatch: parse-or-wait-for-more-bytes, advance the
   buffer only once the whole record is present, run it through the seq/gap
   check, and yield the tagged tuple (or nothing, for records like `HELLO`
   that the reader consumes internally).
4. `Joiner.on_profile_record`'s tag handling for the new tuple shape.
5. A `RecordCodecTest` case covering the encode/decode round trip (and a
   split-feed case if the body has variable-length fields).

Rules:

- Every record's `seq` must flow through `RecordReader`'s gap check
  (`_note_seq`) exactly once, even for record types the reader currently
  discards (e.g. `FRAME_DEF` bodies before a consumer for it exists);
  skipping it desynchronizes gap detection from the rest of the stream.
- A record type without a self-describing length (fixed-size body, or a body
  whose prefix carries the length of what follows) cannot be safely decoded;
  the reader has no way to resynchronize past bytes it cannot parse, so it
  drops the buffer and fails closed (`("gap", None)`) instead of guessing.
- Multi-threaded emitters (the profiling hook runs on every target thread)
  must assign the record's `seq` and write its bytes under the same lock, or
  the pipe can carry seqs out of order and manufacture false gaps.

## Attribution rules

- Use native thread IDs and monotonic nanoseconds for joins.
- Do not use wall time for ordering profile and GPU events.
- Keep the GPU-event hold window unless replacing it with an equally conservative
  ordering strategy.
- Project-file detection must exclude Metagross itself, code with no source
  file (names such as `<string>`), site packages,
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
