"""Release max-div: `make release VERSION=X.Y.Z` runs this folder as `python scripts/release X.Y.Z`.

The release is a list of steps, `RELEASE_STEPS` in runner.py, run in order, phase by phase:

- validation (phase_validation.py) checks the preconditions and writes nothing; `--dry-run` stops here;
- release commit (phase_release_commit.py) builds the release commit and its tag, locally;
- post-release (phase_post_release.py) opens the next development cycle, then pushes main and the
  tag atomically.

Python runs this folder by its path, so `scripts/release/` itself is on `sys.path`, and its modules
import each other by module name alone (`import helpers`).
"""

from runner import main

main()
