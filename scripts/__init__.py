"""Maintainer tooling, never part of the published package.

Each script runs by its path, as `python scripts/<name>.py`. This file makes `scripts/` a package only so that its
tests import as `scripts.tests`, the same layout as `benchmarks/tests/`.
"""
