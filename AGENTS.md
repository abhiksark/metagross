# AGENTS.md

This repository is **Metagross**, a Python/BCC/eBPF tracer that runs one local
Python script and explains which project function caused selected NVIDIA CUDA
driver API calls. Metagross is being productized; keep the implementation small,
deterministic, security-conscious, and usable without BCC/CUDA for ordinary
inspection and unit tests.

## Non-negotiable rules

- Preserve unrelated modified and untracked files; inspect `git status --short`
  before editing.
- Keep unprivileged paths importable and testable without BCC, CUDA, root, or an
  NVIDIA driver.
- Treat live tracing as privileged/security-sensitive code. Be conservative with
  file ownership, symlink handling, dropped credentials, inherited file
  descriptors, environment variables, and target stdout/stderr preservation.
- Preserve target behavior: pass script arguments unchanged, keep target stdout
  unchanged by default, preserve exit statuses/signals, and avoid adding
  Metagross output to stdout.
- Keep trace output schemas stable. Table output must stay human-readable; JSONL
  must stay machine-stable and backwards compatible unless explicitly changed and
  documented.
- Keep attribution conservative. Unknown or ambiguous data should render as
  `<unknown>`/`null`, not a guessed project frame.
- Keep code straightforward: explicit names, small single-purpose functions,
  standard-library-first, and no speculative abstractions.
- Start every new non-empty Python file with its repository-relative path comment,
  matching the existing style.
- Add or update tests for behavior changes and run the smallest meaningful test
  scope before handing off.

## Task-specific guides

Read the smallest relevant set. Code changes normally require Development and
Verification; user-visible/product changes also require Documentation and release.

- [Development guide](docs/agent-guides/development.md): architecture, CLI
  contract, privilege model, eBPF/CUDA tracing, attribution, and productization
  rules.
- [Verification guide](docs/agent-guides/verification.md): targeted unit tests,
  default unprivileged gate, live integration gate, and handoff expectations.
- [Documentation and release guide](docs/agent-guides/docs-and-release.md): README
  update triggers, output schema discipline, examples, packaging considerations,
  and git/release workflow.

## Repository map

- `metagross/__main__.py`: thin `python -m metagross` entry point.
- `metagross/__init__.py`: CLI parsing, validation, privilege handling, forked
  target runner, BCC orchestration, output opening, event loop, and exit behavior.
- `metagross/_bpf.py`: CUDA API table, libcuda discovery, symbol resolution,
  generated eBPF C source, BCC loading, probe attachment, and raw event structs.
- `metagross/_dashboard.py`: private runner for `--web` that serves the
  in-memory dashboard as the unprivileged target user.
- `metagross/_profile.py`: profiling hook installed in the traced child and the
  binary profile-record codec read by the parent.
- `metagross/_viewer.py`: unprivileged streaming JSONL model, summary loading,
  terminal sanitization, and static visual trace rendering.
- `metagross/_follow.py`: bounded incremental JSONL following, partial-line
  handling, trace replacement detection, and final-summary watching.
- `metagross/_tui.py`: dependency-free curses live dashboard and terminal-sized
  overview layout.
- `metagross/_events.py`: profile/GPU stream joining, attribution, kernel-name and
  allocation tracking, detail enrichment, table rendering, and JSONL rendering.
- `tests/test_metagross.py`: unprivileged unit suite plus root/CUDA integration tests
  gated by `RUN_EBPF_INTEGRATION=1`.
- `examples/`: lightweight runnable examples aligned with the README.
