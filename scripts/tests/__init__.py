"""Tests for the maintainer tooling under scripts/.

Deliberately outside the package test suite (tests/): they test no code of the published package, so they run once,
on the default Python, as their own CI job via `make test-scripts`, never inside the package test matrix.
"""
