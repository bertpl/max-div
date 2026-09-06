# Benchmarks vs. 3rd Party Solvers

These pages benchmark `max-div` empirically against other freely available subset-selection
tools. They are the quantitative companion to the qualitative
[Comparison with Other Tools](../comparison.md) page: that page maps the *categories* of
tool and their feature trade-offs; these pages measure how they actually *perform*.

!!! note "Problem naming vintage"
    These pages were measured before the built-in benchmark problems were re-parametrized by n
    and restructured, and they use the problem names and `size` units of that era:

    - their `U1` is the uniform-density problem (now `U2`)
    - their `C2` is the exact-quota problem (now `C1`)
    - sizes are stated as the old `size` parameter (n ≈ 100·size)

    The [current problem definitions](../../solver/test_problems.md) describe today's suite.

The comparison runs in three tiers, by what `max-div` is measured against:

- [**vs. Python Heuristics**](tier2.md) — the single-shot pickers a Python user would
  otherwise reach for (every non-exact tool in the registry). The largest-audience
  comparison, across problem sizes from n = 100 to 100 000.
- [**vs. Exact Solvers**](tier1.md) — CP-SAT, SCIP and HiGHS as an *optimality reference*: how close
  `max-div` gets to a proven optimum, and how far those solvers scale before proving becomes
  intractable.
- [**vs. MDPLIB Best-Known**](tier3.md) — the literature's shared MMDP benchmark: gap to the
  published best-known max-min values on the Glover / Geo / Ran instance sets.

## Shared method

Every tier uses the same method, so numbers are comparable across pages:

- **All tools are scored under `max-div`'s own diversity metrics** (min / mean / geomean
  separation, mean pairwise distance), computed identically for every tool.
- **`max-div` runs a series of wall-clock budgets** (2× steps from 1 ms, `SMART` preset,
  3 seeds); figures plot *measured* solve time, never the nominal budget. Single-shot
  competitors run once per seed. The largest budget differs per tier and is stated on each
  page.
- **One sequential run on one machine** — a 16" MacBook Pro with M3-class CPU.
- Everything is reproducible from the repo-tracked harness under `benchmarks/`.

Across the three tiers, `max-div`:

- leads the Python heuristics at small to moderate selection sizes, and trails the
  farthest-point pickers as the selection grows;
- lands within a few percent of proven optima in milliseconds, where those optima are
  reachable;
- matches or exceeds the literature's best-known values on the larger MMDP instances.

Each page states where `max-div` wins and where it does not.
