[![CI](https://img.shields.io/github/actions/workflow/status/bertpl/max-div/push_to_main.yml?branch=main&label=CI)](https://github.com/bertpl/max-div/actions/workflows/push_to_main.yml)
![coverage](https://img.shields.io/badge/coverage-99.36%25-brightgreen)
![tests](https://img.shields.io/badge/tests-4310-blue)
[![docs-build-status](https://app.readthedocs.org/projects/max-div/badge/?version=latest)](https://max-div.readthedocs.io/en/stable)
[![PyPI](https://img.shields.io/pypi/v/max-div.svg)](https://pypi.org/project/max-div/)
[![Python](https://img.shields.io/pypi/pyversions/max-div.svg)](https://pypi.org/project/max-div/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/bertpl/max-div/blob/main/LICENSE)
[![code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/bertpl/max-div/badge)](https://scorecard.dev/viewer/?uri=github.com/bertpl/max-div)
<p>
  <img src="https://raw.githubusercontent.com/bertpl/max-div/v0.14.1/images/splash_with_version.webp" alt="max-div logo" style="max-width: max(60%, min(100%,800px)); height: auto;">
</p>

# max-div

Configurable solver for **Maximum Diversity Problems** with optional **fairness constraints**:
given `n` items — vectors or precomputed distances — select the `k` most diverse, optionally
subject to per-group minimum/maximum quotas.

max-div is an **anytime heuristic**: it returns a good selection quickly and keeps improving it
for as long as you allow (wall-clock time or iteration count). Two things set it apart among
freely available tools: it is the only
**dedicated diversity solver** with native support for **overlapping fairness constraints**, and
the only one offering a **geometric-mean separation** objective — one of four diversity metrics
it provides (minimum, mean, and geometric-mean separation, plus mean pairwise distance).

<p>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/hero_dark.svg">
    <img src="docs/images/hero_light.svg" alt="Feature comparison of max-div against exact solvers and one-shot pickers: distance metrics, diversity objectives, constraint handling, time budgets and practical scale" style="max-width: max(60%, min(100%,800px)); height: auto;">
  </picture>
</p>

It fills the gap between exact MIP/CP solvers, which prove optimal answers but do not scale, and
single-shot pickers, which are fast but at best have very limited support for constraints. The
[benchmarks](https://max-div.readthedocs.io/en/stable/benchmarks/comparison/overview/) show where
it leads and where it does not.

## Installation

```bash
pip install max-div
```

Python 3.11+; free-threaded builds (3.14t) are supported and CI-tested (see the
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
  max-div relates to exact solvers, greedy pickers, and samplers
- [Benchmarks](https://max-div.readthedocs.io/en/stable/benchmarks/comparison/overview/) — the
  measured comparison against third-party tools

## License

Licensed under the [Apache License 2.0](https://github.com/bertpl/max-div/blob/main/LICENSE).
