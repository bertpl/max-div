[![CI](https://img.shields.io/github/actions/workflow/status/bertpl/max-div/push_to_main.yml?branch=main&label=CI)](https://github.com/bertpl/max-div/actions/workflows/push_to_main.yml)
![coverage](https://img.shields.io/badge/coverage-99.26%25-brightgreen)
![tests](https://img.shields.io/badge/tests-4767-blue)
[![docs-build-status](https://app.readthedocs.org/projects/max-div/badge/?version=latest)](https://max-div.readthedocs.io/en/stable)
[![PyPI](https://img.shields.io/pypi/v/max-div.svg)](https://pypi.org/project/max-div/)
[![Python](https://img.shields.io/pypi/pyversions/max-div.svg)](https://pypi.org/project/max-div/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/bertpl/max-div/blob/main/LICENSE)
[![code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/bertpl/max-div/badge)](https://scorecard.dev/viewer/?uri=github.com/bertpl/max-div)
<p align="center">
  <img src="https://raw.githubusercontent.com/bertpl/max-div/v0.17.3/images/splash_with_version.webp" alt="max-div logo" style="max-width: max(60%, min(100%,900px)); height: auto;">
</p>

# max-div

**A versatile, high-performance solver for Maximum Diversity Problems** — select the `k` most
diverse of `n` items, under optional fairness constraints.

## Highlights

- ⚡ obtains **near-optimal results within seconds-to-one-minute** for problems up to `n=200k`

- ⏱️ runs within an **arbitrary solve budget** — wall-clock time or iteration count

- 🚀 leverages [numba](https://pypi.org/project/numba/) **JIT-compilation for maximum speed** without relying on pre-compiled binaries

- ⚖️ natively supports flexible **fairness constraints**

    - uniquely supports **constraints with overlapping groups & ranged counts**

    - returns the **least infeasible solution** (with configurable weighted linear or quadratic penalties) when constraints conflict

    - provides **proofs of (in)feasibility**

- 📐 uniquely supports **5+ distance metrics** (L1, L2, L∞, Minkowski, cosine — or precomputed distances) and **4 diversity metrics** (minimum, mean & geomean separation + mean pairwise distance) in any combination

- 💾 computes item distances **eagerly when memory allows** (maximum speed), **lazily when problem size requires** (minimal memory usage)

- 🤝 leverages multi-core CPUs with **parallel workers** in independent, cooperative or dynamically grouped configurations, **without duplicating core problem data**

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/hero_dark.svg">
    <img src="docs/images/hero_light.svg" alt="Feature comparison of max-div against exact solvers and one-shot pickers: distance metrics, diversity objectives, constraint handling, time budgets and practical scale" style="max-width: max(60%, min(100%,1000px)); height: auto;">
  </picture>
</p>

The [benchmarks](https://max-div.readthedocs.io/en/stable/benchmarks/comparison/overview/)
compare max-div in depth with 10 other freely available solvers.

## Installation

```bash
pip install max-div
```

Python 3.12+; free-threaded builds (3.14t) are supported and CI-tested (see the
[installation notes](https://max-div.readthedocs.io/en/stable/getting_started/#installation) for
the numba version they require).

## Quick start

```python
import numpy as np
from max_div import MaxDivProblem, MaxDivSolverBuilder, seconds

rng = np.random.default_rng(42)
vectors = rng.random((200, 5))               # 200 points in 5 dimensions

# select the 20 most diverse, improving for up to 5 seconds
problem = MaxDivProblem.new(vectors, k=20)
solution = MaxDivSolverBuilder(problem).with_preset(seconds(5)).build().solve()

print(solution.i_selected)                   # indices of the selected items
```

### With fairness constraints

Require a minimum and/or maximum number of selected items from given subsets — useful for fair
representation across groups. Groups may overlap, and infeasible constraints degrade gracefully
to the least-infeasible selection rather than failing.

```python
from max_div import Constraint

# require between 8 and 12 of the selected items from each half of the data
constraints = [
    Constraint(int_set=set(range(0, 100)),   min_count=8, max_count=12),
    Constraint(int_set=set(range(100, 200)), min_count=8, max_count=12),
]
problem = MaxDivProblem.new(vectors, k=20, constraints=constraints)
```

## Documentation

Full documentation lives at **[max-div.readthedocs.io](https://max-div.readthedocs.io)**,
including:

- [Getting started](https://max-div.readthedocs.io/en/stable/getting_started/) — installation,
  distance and diversity metrics, solver presets
- [Comparison with other tools](https://max-div.readthedocs.io/en/stable/comparison/) — how
  max-div relates to exact solvers, greedy pickers, clustering and samplers
- [Benchmarks](https://max-div.readthedocs.io/en/stable/benchmarks/comparison/overview/) — the
  measured comparison against third-party tools

## License

Licensed under the [Apache License 2.0](https://github.com/bertpl/max-div/blob/main/LICENSE).
