# Diversity & Distance

## From Items to Diversity

The solver evaluates how diverse a selection is through a three-step process:

```
Items  ──>  Pairwise Distances  ──>  Contributions  ──>  Diversity Score
  (n)         (n*(n-1)/2)               (k x 1)              (scalar)
```

1. **Pairwise distances** are computed once upfront between all `n` [items](glossary.md#item) using the chosen
   distance metric. These are stored as a condensed distance vector (like scipy's `pdist`).

2. A **diversity contribution** is computed for each selected item -- a per-item quantity where
   higher means "contributes more diversity". Which quantity that is depends on the diversity metric's
   family:

    - **Separation family** (all `*_SEPARATION` metrics): the distance to the item's
      *nearest neighbor* within the current selection. An item with high separation is
      well-spread from the rest of the selection; an item with low separation is close to at
      least one other selected item.
    - **Mean-distance family** (`MEAN_PAIRWISE_DISTANCE`): the item's *mean distance to the
      other selected items* -- exactly how much the item contributes to the total spread
      of the selection.

3. The **diversity score** aggregates the `k` contribution values into a single scalar using the
   chosen diversity metric.

## Distance Metrics

The distance metric determines how the distance between two vectors is measured.

| Metric | Formula | Notes |
|--------|---------|-------|
| `l2_euclidean()` | $$d = \sqrt{\sum_i (x_i - y_i)^2}$$ | Standard Euclidean distance. Default. |
| `l1_manhattan()` | $$d = \sum_i \lvert x_i - y_i \rvert$$ | Less sensitive to outlier dimensions. |
| `l2s_euclidean_squared()` | $$d = \sum_i (x_i - y_i)^2$$ | Avoids the square root. Produces identical solutions to `l2_euclidean()` when used with `GEOMEAN_SEPARATION`, since the geometric mean preserves distance ordering regardless of squaring. |
| `linf_chebyshev()` | $$d = \max_i \lvert x_i - y_i \rvert$$ | Chebyshev distance: set by the single largest per-dimension gap, so no dimension's difference is averaged away by the others. |
| `cosine()` | $$d = 1 - \frac{x \cdot y}{\lVert x \rVert \, \lVert y \rVert}$$ | Angular distance in $[0, 2]$, invariant to vector magnitude -- the natural choice for embedding-style vectors. Undefined for zero vectors, which are rejected at problem construction. |
| `minkowski(p, root=True)` | $$d = \Big( \sum_i \lvert x_i - y_i \rvert^p \Big)^{1/p}$$ | The general family behind `l1_manhattan()` ($p=1$), `l2_euclidean()` ($p=2$) and `linf_chebyshev()` ($p=\infty$); any $p > 0$ is accepted, and those special values resolve to the dedicated metrics. |

- **Speed depends on `p`.** The values $p \in \{1, 2, \infty, 0.5, 0.25, 0.125\}$ compute with hardware arithmetic; every other $p$ pays a `pow` call per dimension, well over an order of magnitude more per term.
- **`root=False` skips the outer $1/p$ root**, exactly as `l2s_euclidean_squared()` does for `l2_euclidean()` -- see that row above.
- **For $0 < p < 1$ the `root=True` form violates the triangle inequality** and is not a strict metric, while the `root=False` form is one -- the solver never relies on the triangle inequality, so both are usable.

## Separation

The **[separation](glossary.md#separation)** of a selected item is its minimum distance to any
other selected item:

$$\text{sep}(v) = \min_{u \in S,\; u \neq v} \; d(v, u)$$

where $S$ is the current selection and $d$ is the chosen distance metric.

Separation is maintained incrementally during optimization:

- **Adding** an item only requires checking its distances to all other items and
  updating any separations that decrease.
- **Removing** an item requires recomputing separation only for items whose nearest
  neighbor was the removed one.

This incremental update is much cheaper than recomputing all separations from scratch after
each swap.

## Mean Distance

The **mean distance** of a selected item is its mean distance to the other selected items:

$$\text{md}(v) = \frac{1}{k - 1} \sum_{u \in S,\; u \neq v} d(v, u)$$

It is maintained incrementally as well, and even more cheaply than separation: adding an item
adds one distance to every item's running sum, and removing one subtracts it exactly -- no
rescan of the selection is ever needed.

## Diversity Metrics

Each diversity metric aggregates the `k` separation values differently. (For how these objectives
map onto the operations-research literature -- p-dispersion, Max-SumMin, classical MaxSum MDP --
see [Objectives & the Diversity-Problem Landscape](objectives.md).)

| Metric | Formula | Characteristics |
|--------|---------|-----------------|
| `GEOMEAN_SEPARATION` | $$\exp\!\left(\frac{1}{k}\sum_{v \in S} \ln(\text{sep}(v))\right)$$ | **Default.** Balances all separations. Sensitive to any item with low separation -- a single poorly-placed item drags down the score. |
| `MIN_SEPARATION` | $$\min_{v \in S} \text{sep}(v)$$ | Only considers the worst-off item (the closest pair). Equivalent to the *p-dispersion* problem. Many swaps produce tied scores. |
| `MEAN_SEPARATION` | $$\frac{1}{k}\sum_{v \in S} \text{sep}(v)$$ | Averages all separations. Less sensitive to individual outliers than geomean, but can be dominated by a few very high separations. |
| `APPROX_GEOMEAN_SEPARATION` | Same as geomean but using fast log/exp approximations | Slightly less accurate but faster per iteration. Useful for large-scale problems where iteration speed matters more than per-iteration precision. |
| `MEAN_PAIRWISE_DISTANCE` | $$\frac{2}{k(k-1)}\sum_{\{u,v\} \subseteq S} d(u, v)$$ | Mean distance over all selected *pairs* -- the classical **max-sum diversity** objective (MaxSum MDP, also known as *remote-clique*). Maximizes total spread: selections gravitate to the outer regions of the data, and near-duplicates are tolerated if both sit far from everything else. |

### Which metric to choose?

- **`GEOMEAN_SEPARATION`** is the best default. It naturally penalizes any clustering in the
  selection while remaining smooth and differentiable in most of the search space.
- **`MIN_SEPARATION`** is appropriate when you specifically care about the worst-case nearest
  neighbor distance (e.g., facility placement where minimum coverage radius matters).
  Expect slower convergence due to many tied scores.
- **`MEAN_SEPARATION`** maximizes total spread. It may tolerate some clustering as long as
  other items compensate with large separations.
- **`APPROX_GEOMEAN_SEPARATION`** is a drop-in replacement for `GEOMEAN_SEPARATION` when
  you want to trade a small amount of precision for more iterations per second.
- **`MEAN_PAIRWISE_DISTANCE`** is the objective to pick when you want classical max-sum
  diversity semantics ("maximize total spread") or want results comparable with the MaxSum
  MDP literature. Unlike every separation metric it does *not* penalize near-duplicates per
  se -- two nearly identical items at the data's boundary can both be kept. Note that the
  solver's swap heuristics were originally designed and tuned around separation contributions; they
  are correct for this metric (its per-item contribution is the item's exact marginal
  contribution to the objective), but the separation metrics remain the most battle-tested
  choice.
