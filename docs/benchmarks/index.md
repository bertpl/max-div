# Benchmark Results

This section contains three categories of benchmark results:

- [**3rd Party Solvers**](./comparison/overview.md) — `max-div` against other freely available
  subset-selection tools: Python heuristics, exact MIP/CP solvers, and the MDPLIB
  best-known values from the literature.
- [**Solver Presets**](./solver/test_problems.md) — `max-div`'s own initialization/optimization
  strategies and presets measured against each other on the built-in test problems.
- [**Internal Primitives**](./internal/overview.md) — micro-benchmarks of internal building
  blocks (diversity-metric kernels, random samplers, probability selectivity).

The Solver-Preset and Internal-Primitive benchmarks are triggerable via the CLI:
```
max-div benchmark <category> <sub-command>
```

All benchmarks are run on a 16" MacBook Pro with M3-class CPU.
