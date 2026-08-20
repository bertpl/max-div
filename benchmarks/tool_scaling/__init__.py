"""Tool-scaling benchmarks: measure each tool's memory / time / quality tool-scaling values.

The published capability table carries three measured scaling limits per tool (see the docs'
*Capability Definitions* page for their exact definitions and the measurement protocol).
This package is the machinery that measures them: the candidate-size grid, the per-tool
run configurations, a subprocess runner that enforces the time budget and the memory cap,
and the stage drivers that turn raw run records into limit values.
"""
