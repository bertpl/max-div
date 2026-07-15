"""Tests for the comparison-benchmark harness.

Deliberately outside the package test suite (tests/): these need the `benchmarks`
dependency group and run as their own CI job (single Python, numba on) via
`make test-benchmarks`, never inside the package test matrix.
"""
