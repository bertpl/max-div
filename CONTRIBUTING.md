# Contributing to max-div

Thanks for your interest in contributing.

## Security

Found a vulnerability? Please report it privately — see
[`SECURITY.md`](SECURITY.md). Do not open a public issue for it.

## Dev setup

One-time setup on a fresh clone:

```bash
make dev-setup
```

This syncs dev dependencies via `uv` and installs pre-commit hooks.

## Common commands

```bash
make test            # Run the test suite (pytest)
make format          # Format and auto-fix with ruff
make lint            # Run the full pre-commit hook suite over all files
make coverage        # Run the test suite with an HTML coverage report
```

## Branching

Branch names follow the pattern:

```
<prefix>/<short-slug>
```

- **Prefix** — one of `feat/`, `fix/`, `chore/`, `docs/`, `refactor/`,
  `test/`. CI rejects anything else.
- **Slug** — short kebab-case description, lowercase letters, digits,
  and hyphens only. When a GitHub issue exists, start the slug with its
  number (`feat/42-...`).

Examples: `chore/oss-scaffolding`, `feat/07-new-input-format`,
`fix/12-solver-state-crash`.

## Pull requests

PRs are merged into `main` via **squash merge only** (repo settings
disable merge commits and rebase merges). Each PR therefore produces
exactly one commit on `main`. The squash commit subject is the PR
title and the body is the PR body, so write both with care — they
become the permanent history. The feature branch is deleted
automatically on merge.

## Commit messages

Subject line uses the same short-form prefixes as branches:

```
<prefix>: <imperative summary>
```

- **Prefix** — `feat`, `fix`, `chore`, `docs`, `refactor`, `test`
  (matching the branch prefix is the common case but not required).
- **Summary** — imperative mood, lowercase, no trailing period,
  ideally under 72 characters.

Examples:

```
feat: add fairness-constraint presets
fix: handle empty candidate set
chore: bump numba floor
docs: clarify separation-metric semantics
```

The body (optional) explains *why*, not *what*. Wrap at ~72 characters.

## Changelog

Add an entry under the appropriate category in the `## Unreleased` section
of [`CHANGELOG.md`](CHANGELOG.md) as part of your PR. CI requires this for
`feat/` and `fix/` branches.

Changelog entries are **user-facing** — write them for someone deciding
whether to upgrade, not for someone reviewing the implementation. Focus on
what changed from the user's perspective.

**Keep each entry to a single line.** Avoid verbosity; omit internal details
(class names, wiring, refactors that don't affect behavior). Expand to a
second line only when a single line genuinely can't convey what the change
is about.
