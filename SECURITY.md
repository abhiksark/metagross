# Security policy

Only the latest public-preview commit is supported. There are no supported
stable versions or backport branches. Updates are delivered through the source
repository; follow the latest changes before running a privileged capture.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/abhiksark/metagross/security/advisories/new).
Do not open a public issue containing exploit details, credentials, private URLs,
or unsanitized traces. Include the affected commit, host and Python versions,
reproduction steps, expected and observed behavior, and the privilege boundary
involved. Share only the minimum sanitized evidence needed to reproduce it.

If the private reporting form is unavailable, request a private reporting channel
without publishing vulnerability details. There is no promised response deadline.

## Scope and trust model

Metagross is a local diagnostic for trusted scripts. The controller is privileged;
under validated sudo metadata the target runs as the invoking user. Direct root
execution also runs the target as root. Neither tracing nor the Docker examples
sandbox untrusted code. A compromised invoking account, root process, or browser
is outside the protection offered by the viewer authentication boundary.

The loopback dashboard requires a private viewer bearer token to read trace
state. The separate producer token authorizes capture delivery only. Keep both
private. Do not expose the server through a proxy or port forward. Trace files
can contain source paths, function/kernel names, handles, and timing information.
Review [file safety and the trust boundary](docs/reference.md) before sharing
captures or changing privileged code.
