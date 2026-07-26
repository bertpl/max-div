# Scoring

## Multi-Component Score

Every selection (intermediate or final) is evaluated by a `Score` with four components, in
**strict priority order**:

| Component | Range | Meaning |
|-----------|-------|---------|
| `size` | 0 to 1 | 1.0 when exactly `k` [items](glossary.md#item) are selected. Used internally during initialization; always 1.0 in a final solution. |
| `constraints` | 0 to 1 | 1.0 when all fairness constraints are satisfied. Lower values indicate more constraint violations. |
| `diversity` | 0+ | The main diversity metric value (higher is better). |
| `div_tie_breakers` | 0+ | Secondary diversity metrics, used to break ties. |

When comparing two solutions, the solver checks components in priority order: a solution with
higher `constraints` score always beats one with higher `diversity`, regardless of how large the
diversity difference is. Within the same constraint satisfaction level, higher diversity wins.

## Why Priority Ordering?

Consider a problem with fairness constraints. Without priority ordering, the solver might find
a highly diverse selection that badly violates constraints. The strict ordering ensures that:

- The solver always moves towards feasible solutions first
- Diversity is only optimized within the space of equally-feasible solutions
- Infeasible problems gracefully degrade to the least-violated solution

## Diversity Tie-Breakers

Some diversity metrics produce many tied scores, which makes it hard for swap-based optimization
to make progress. Tie-breakers are additional diversity metrics that help distinguish between
solutions with equal primary diversity scores.

The solver automatically selects appropriate tie-breakers based on your chosen diversity metric:

| Primary Metric | Default Tie-Breakers | Why |
|---------------|---------------------|-----|
| `MIN_SEPARATION` | `APPROX_GEOMEAN_SEPARATION`, `NON_ZERO_SEPARATION_FRAC` | Min-separation only depends on the closest pair. Swapping any other item doesn't change the score, causing many ties. The geomean tie-breaker guides the solver towards improving the overall spread. |
| `GEOMEAN_SEPARATION` | `NON_ZERO_SEPARATION_FRAC` | Geomean is zero if any separation is zero. When multiple separations are zero, single swaps can't improve the score. The non-zero fraction tie-breaker guides the solver towards eliminating zero-distance items. |
| `MEAN_SEPARATION` | *(none)* | Mean separation rarely produces ties. |

You can override the defaults via `MaxDivSolverBuilder.with_diversity_tie_breakers()`.

## Soft Constraints (Advanced)

By default, constraints are **hard**: constraint score strictly outranks diversity. The solver's
optimization strategies support a `constraint_softness` parameter (0.0 to 1.0) that blends
diversity into the constraint score:

- `0.0` -- fully hard constraints (default)
- `0.0 < s < 1.0` -- partially soft: `constraint_score^(1-s) * diversity^s`
- `1.0` -- fully soft: constraints are ignored, only diversity matters

This is configured at the strategy level and is primarily useful for advanced solver tuning.
The built-in presets use hard constraints with a brief initial phase where infeasible diversity
is ignored, allowing the solver to focus on reaching feasibility before optimizing diversity.
