# Documentation and release guide

Metagross is intended to become a product. Keep user-facing documentation and
release behavior accurate whenever behavior changes.

Keep the README focused on the first local capture and product boundaries.
Detailed CLI, schema, API, file-safety, and viewer contracts belong in
`docs/reference.md`; contributor gates belong in `CONTRIBUTING.md` and the
verification guide. Update those alongside affected README examples without
duplicating the full reference in the README.

## Documentation update triggers

Update `docs/reference.md` in the same change, and `README.md` where it
mentions the behavior, when altering:

- CLI flags, option ordering, validation, usage, or exit behavior.
- Required OS, kernel, Python, BCC, CUDA, or NVIDIA driver assumptions.
- Privilege model, sudo behavior, credential dropping, or target environment.
- Output file creation, ownership, truncation, or safety semantics.
- Table columns or detail formatting.
- JSONL top-level fields or detail fields.
- Traced CUDA API coverage.
- Attribution behavior, limitations, overhead, or known blind spots.
- Visual viewer commands, layouts, supported schemas, or TTY requirements.
- Integration-test commands or setup instructions.

## Schema/version discipline

- Treat JSONL output as a product interface.
- Prefer additive fields over renaming or removing fields.
- Use `null` for unknown JSON attribution fields.
- Keep table output stable enough for humans; do not optimize it for parsing at
  the expense of readability.
- If a breaking schema change becomes necessary, document it in the
  reference and the changelog, and update the tests.

## Examples

- Keep `examples/README.md` and example scripts synchronized with top-level usage.
- Examples should be small, local, and easy to inspect.
- Do not require network access from examples unless the product explicitly adds a
  network-backed feature.
- Avoid making examples depend on heavyweight frameworks unless the example is
  specifically about that framework.

## Packaging checklist

When changing packaging or distribution, keep these documented and tested:

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
- Write the subject as `type: summary`: imperative, lowercase after the colon,
  at most 72 characters, no trailing period.
- Add a body only when the reason is not obvious from the subject: at most four
  lines wrapped at 72, saying why rather than how. Do not reference internal
  findings, plans, or files that are not in the repository.
- Make one logical change per commit. Fold a fix to the commit it repairs
  before pushing instead of adding a follow-up commit.
- Update `CHANGELOG.md` for user-visible changes.
- The version lives in `metagross/__init__.py` (`__version__`); `pyproject.toml`
  reads it from there.

## Versions

- A release is `MAJOR.MINOR.PATCH`, set by a `chore: release X.Y.Z` commit that
  also dates the changelog section, and carries the annotated tag `vX.Y.Z`.
- Before 1.0, a patch release fixes behavior and may add options, output
  fields, or summary fields. A minor release may change or remove them; the
  changelog entry then starts with `Breaking:`.
- Right after tagging, set `__version__` on master to the next patch version
  with a `.dev0` suffix and open an `Unreleased` changelog section, so a
  checkout between releases never reports a released version.
- The Python API is the names in `metagross.__all__`. Other names in the
  package are internal, with or without a leading underscore.
