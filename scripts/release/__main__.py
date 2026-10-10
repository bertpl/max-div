"""Release max-div: `make release VERSION=X.Y.Z` runs this folder as `python scripts/release X.Y.Z`.

The release is a list of steps, `RELEASE_STEPS` in release_runner.py, run in order in 3 phases:

- validation (validation.py) checks the preconditions and writes nothing; `--dry-run` stops here;
- release commit (release_commit.py) bumps the version, finalizes the changelog, stamps the README
  badges and the splash, commits the release and tags it;
- post-release (post_release.py) opens a fresh `## Unreleased` section, commits it, and pushes main
  and the tag atomically.

The folder runs by its path, so its modules import each other by bare name; a test loads them with
`scripts.tests.helpers.load_release_module`.
"""

from release_runner import main

main()
