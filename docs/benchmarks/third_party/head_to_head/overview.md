# Benchmarks vs. 3rd Party Solvers

These pages measure `max-div` against other freely available subset-selection tools. They are the quantitative companion to the qualitative [Comparison with Other Tools](../comparison.md) page: that page maps the *categories* of tool and their feature trade-offs; these pages measure how they perform on the same problems.

The comparison runs in three tiers, by what `max-div` is measured against:

- [**vs. Exact Solvers**](tier1.md) — CP-SAT, SCIP and HiGHS as the optimality reference: how close `max-div` gets to a proven optimum, and how far each solver proves before its cap.
- [**vs. Python Heuristics**](tier2.md) — the one-shot pickers a Python user would otherwise reach for: at what budget `max-div` passes the best of them, per problem size.
- [**vs. MDPLIB Best-Known**](tier3.md) — the literature's shared MMDP instances: how far `max-div` is from the best published value per instance, as a function of budget.

## Shared method

All three tiers share these settings:

- the 60 s time budget, the 1-2-5 size grid and the reference machine of the [solver-scaling protocol](../scaling/protocol.md), and the [solver configurations](../scaling/solver_configs.md) measured there;
- 3 seeds per cell;
- `max-div` in its `DEFAULT` preset, solved once per time budget of a series of increasing budgets, once with one worker and once with 12 workers; each solve is timed end to end;
- every tool's selection evaluated under identical criteria.

Everything is reproducible from the harness under `benchmarks/`.
