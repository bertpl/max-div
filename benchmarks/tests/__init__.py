"""Tests for the comparison-benchmark harness.

Deliberately outside the package test suite (tests/): these need the `benchmarks`
dependency group and run once, on a single Python with numba on, via
`make test-benchmarks`, never inside the package test matrix.
"""
