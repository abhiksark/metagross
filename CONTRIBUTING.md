# Contributing

Metagross is an experimental source-only preview. Use issues for reproducible
bugs and concrete feature requests, and pull requests for focused changes.
For security concerns, follow [SECURITY.md](SECURITY.md) instead of opening a
public issue. Avoid posting unsanitized workload traces or private viewer URLs.

## Local setup

```sh
git clone https://github.com/abhiksark/metagross.git
cd metagross
/usr/bin/python3 -m metagross --help
/usr/bin/python3 -m unittest -v
```

Use Python 3.10 or newer. Ordinary inspection and unit tests need only the
standard library, with no root, BCC, CUDA, or GPU. Optional Node.js tests exercise
the browser bootstrap when Node.js is present; it is not a runtime dependency.
Users run Metagross through Docker; see the [quick start](README.md#quick-start).
The privileged live test gate below runs on the host instead and needs the
system Python with BCC bindings and headers for the running kernel. On Ubuntu:
`sudo apt install python3-bpfcc "linux-headers-$(uname -r)"`.

## Changes and pull requests

Read [AGENTS.md](AGENTS.md) and the relevant
[development guide](docs/agent-guides/development.md) before changing behavior.
Preserve unrelated work. Keep functions small, names explicit, and dependencies
minimal. Follow the Zen of Python and Google Python style. Begin each new
non-empty Python file with its repository-relative path comment.

Add or update focused tests for behavior changes. Keep unprivileged modules
importable without BCC or CUDA. Preserve target arguments, stdout/stderr, exit
behavior, conservative attribution, and trace schemas. Update the README for
user-facing changes and the [reference](docs/reference.md) for detailed contracts.

Describe the problem, resulting behavior, and checks run in the pull request.
Include sanitized reproduction steps and relevant host versions for tracer
failures. State explicitly when privileged verification was not run. Do not
include assistant attribution in commit messages.

## Verification

Run the smallest affected test first, then the default unprivileged gate:

```sh
/usr/bin/python3 -m unittest -v
git diff --check
```

CI also runs `ruff check .` (ruff 0.16.8) for undefined names and unused
imports; run it locally if you have ruff installed.

CI also runs the tracer against a stub driver library, with root and BCC but no
GPU; see the stub live gate in the
[verification guide](docs/agent-guides/verification.md).

Changes to live tracing, probe attachment, credentials, output ownership, or
cleanup also need the privileged gate on a compatible host with BCC and NVIDIA
driver support:

```sh
sudo env RUN_EBPF_INTEGRATION=1 \
  /usr/bin/python3 -m unittest -v tests.test_metagross.LiveTraceTest
```

This executes local GPU work and installs temporary BPF probes. Review the code
and run it only on a host you control. See the
[verification guide](docs/agent-guides/verification.md) for focused tests,
direct-delivery failures, real-browser checks, and cleanup expectations.
Passing the unprivileged suite alone does not establish live CUDA correctness.
