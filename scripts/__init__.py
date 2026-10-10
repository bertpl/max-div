"""This folder holds maintainer tooling that is never part of the published package.

Each script, and each script folder such as `release/`, is run or loaded by its file path, never imported as
`scripts.<name>`. A script imports another script as top-level module `<name>`, and that import fails when the
importing script was loaded as `scripts.<name>`.

A test loads a script with `scripts.tests.helpers.load_script`, and a module of `release/` with
`scripts.tests.release.helpers.load_release_module`. This file makes `scripts/` a package only so that its tests
import as `scripts.tests`, the same layout as `benchmarks/tests/`.
"""
