"""This folder holds maintainer tooling that is never part of the published package.

Each script is run or loaded by its file path, never imported as `scripts.<name>`, because a script imports another
script as top-level module `<name>`, which does not find it as `scripts.<name>`; a test loads a script with
`scripts.tests.helpers.load_script`. This file makes `scripts/` a package only so
that its tests import as `scripts.tests`, the same layout as `benchmarks/tests/`.
"""
