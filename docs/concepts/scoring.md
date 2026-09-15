# Scoring

## Multi-Component Score

Every selection (intermediate or final) is evaluated by a `Score` with four components, in
**strict priority order**:

| Component | Range | Meaning |
|-----------|-------|---------|
| `size` | 0 to 1 | 1.0 when exactly `k` [items](glossary.md#item) are selected. Used internally during initialization; always 1.0 in a final solution. |
| `constraints` | 0 to 1 | 1.0 when all fairness constraints are satisfied. Lower values indicate more constraint violations. |
| `diversity` | 0+ | The main diversity metric value (higher is better). |
| tie-breakers | 0+ | Secondary diversity metrics, used to break ties. |

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

Two selections with the same primary diversity score are tied. A tie is a problem only when the
swaps that lead toward the optimum are among the tied ones: the solver then has no incentive to
take them, or cannot leave a plateau at all. Tie-breakers are additional diversity metrics that
rank tied selections, so that those swaps score higher.

The solver automatically selects appropriate tie-breakers based on your chosen diversity metric:

| Primary Metric | Default Tie-Breakers | Why |
|---------------|---------------------|-----|
| `MIN_SEPARATION` | `APPROX_GEOMEAN_SEPARATION`, `NON_ZERO_SEPARATION_FRAC` | The score depends on the closest pair alone, so a swap that spreads the other items leaves it unchanged. Such a swap has value, though: it frees room around the closest pair and makes a later swap that moves one of its items apart more likely. The approximate geomean rewards it; the non-zero fraction takes over once that geomean has underflowed to zero. |
| `GEOMEAN_SEPARATION`, `APPROX_GEOMEAN_SEPARATION`, `HARMONIC_MEAN_SEPARATION` | `NON_ZERO_SEPARATION_FRAC` | These means are zero as soon as one separation is zero. Once two or more pairs coincide, no single swap moves the score off zero, so the solver would be stuck. The non-zero fraction counts the coincident pairs down until the score is non-zero again. |
| `MEAN_SEPARATION`, `MEAN_PAIRWISE_DISTANCE` | *(none)* | Every swap that changes a separation changes the score, so a tie never hides an improvement. |

A [hybrid diversity metric](../reference/metrics/HybridDiversityMetric.md) follows the same rule over its terms' metrics:

- the approximate geomean is needed when any term is min-separation;
- the non-zero fraction is needed when any term is min-separation or is zero when any separation is zero.

Each tie-breaker then applies to every distance that the hybrid's terms use, aggregated over those distances by its own mean:

- the approximate geomean by a geometric mean;
- the non-zero fraction by an arithmetic mean, so a distance with no non-zero separation lowers the tie-breaker without making it zero.

You can override the defaults via `MaxDivSolverBuilder.with_diversity_tie_breakers()`, except for a hybrid, whose tie-breakers cannot be overridden.

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
