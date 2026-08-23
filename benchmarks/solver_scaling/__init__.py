"""Solver-scaling benchmarks: how far each solver scales in memory and time.

The package measures, per solver configuration, the largest problem size n it still handles
under a memory cap (`memory_fit`) and within a time budget (`time_stage`). Every run executes in
its own subprocess (`runner`, `run_one`) so a stuck solver can be killed and its peak memory is
its own; `configs` holds the per-solver run configurations and `grid` the shared size grid and
budget/memory constants. Published on the third-party solver-scaling pages; the measurement
protocol is `docs/benchmarks/third_party/scaling/protocol.md`.
"""
