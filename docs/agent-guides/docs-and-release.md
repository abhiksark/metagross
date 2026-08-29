# Documentation and release guide

Metagross is intended to become a product. Keep user-facing documentation and
release behavior accurate whenever behavior changes.

## README update triggers

Update `README.md` in the same change when altering:

- CLI flags, option ordering, validation, usage, or exit behavior.
- Required OS, kernel, Python, BCC, CUDA, or NVIDIA driver assumptions.
- Privilege model, sudo behavior, credential dropping, or target environment.
- Output file creation, ownership, truncation, or safety semantics.
- Table columns or detail formatting.
- JSONL top-level fields or detail fields.
- Traced CUDA API coverage.
- Attribution behavior, limitations, overhead, or known blind spots.
- Integration-test commands or setup instructions.

## Schema/version discipline

- Treat JSONL output as a product interface.
- Prefer additive fields over renaming or removing fields.
- Use `null` for unknown JSON attribution fields.
- Keep table output stable enough for humans; do not optimize it for parsing at
  the expense of readability.
- If a breaking schema change becomes necessary, document it clearly in the
  README and tests.

## Examples

- Keep `examples/README.md` and example scripts synchronized with top-level usage.
- Examples should be small, local, and easy to inspect.
- Do not require network access from examples unless the product explicitly adds a
  network-backed feature.
- Avoid making examples depend on heavyweight frameworks unless the example is
  specifically about that framework.

## Productization checklist for future packaging

When packaging or distribution is introduced, make sure to document and test:

- Supported Python versions.
- Supported Linux distributions and kernel versions.
- BCC installation expectations.
- NVIDIA driver/CUDA compatibility expectations.
- Whether the product requires `/usr/bin/python3` or supports arbitrary Python
  interpreters.
- How root/sudo is required and how target privileges are dropped.
- Whether generated trace files contain sensitive file paths, function names,
  kernel names, or timing data.

## Git and release workflow

- Check `git status --short` before editing and before handoff.
- Keep changes focused and reviewable.
- Do not commit, tag, or push unless explicitly asked.
- If asked to commit, use Conventional Commit style, for example:
  - `fix: preserve target stderr on child exit`
  - `feat: trace cuda async allocation calls`
  - `test: cover output symlink rejection`
  - `docs: document live integration requirements`
- If a changelog is added later, update it for user-visible changes.
- If semantic releases are added later, ensure commit messages match the release
  tooling's expectations.
