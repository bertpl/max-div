# Diversity & distance

## I. From items to diversity { #from-items-to-diversity }

The solver evaluates how diverse a selection is through a three-step process:

```
Items  ──>  Pairwise Distances  ──>  Contributions  ──>  Diversity Score
  (n)         (n*(n-1)/2)               (k x 1)              (scalar)
```

1. **Pairwise distances** are computed once upfront between all `n` [items](glossary.md#item) using the chosen
   distance metric. These are stored as a full distance matrix, or computed on demand when the
   matrix would not fit in memory (see [distance storage](distance_storage.md)).

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

## II. Distance metrics { #distance-metrics }

The distance metric determines how the distance between two vectors is measured.

| Metric | Formula | Notes |
|--------|---------|-------|
| `l2_euclidean()` | $d = \sqrt{\sum_i (x_i - y_i)^2}$ | Standard Euclidean distance. Default. |
| `l1_manhattan()` | $d = \sum_i \lvert x_i - y_i \rvert$ | Less sensitive to outlier dimensions. |
| `l2s_euclidean_squared()` | $d = \sum_i (x_i - y_i)^2$ | Avoids the square root. Produces identical solutions to `l2_euclidean()` when used with `GEOMEAN_SEPARATION`, since the geometric mean preserves distance ordering regardless of squaring. |
| `linf_chebyshev()` | $d = \max_i \lvert x_i - y_i \rvert$ | Chebyshev distance: set by the single largest per-dimension gap, so no dimension's difference is averaged away by the others. |
| `l_minus_inf()` | $d = \min_i \lvert x_i - y_i \rvert$ | The L−∞ distance: the smallest per-dimension gap, the $p \to -\infty$ end of the power-mean family whose $p \to +\infty$ end is `linf_chebyshev()`. The distance between two points is the gap in the coordinate projection where they are closest, so a selection kept apart under it is spread in every coordinate projection; `geometric_mean()` is its smoothed form. A shared coordinate makes the distance zero. |
| `cosine()` | $d = 1 - (x \cdot y) \,/\, (\lVert x \rVert \, \lVert y \rVert)$ | Angular distance in $[0, 2]$, invariant to vector magnitude -- the natural choice for embedding-style vectors. Undefined for zero vectors, which are rejected at problem construction. |
| `minkowski(p, root=True)` | $d = \Big( \sum_i \lvert x_i - y_i \rvert^p \Big)^{1/p}$ | The general family behind `l1_manhattan()` ($p=1$), `l2_euclidean()` ($p=2$) and `linf_chebyshev()` ($p=\infty$); any $p > 0$ is accepted, and those special values resolve to the dedicated metrics. |
| `geometric_mean()` | $d = \Big( \prod_i \lvert x_i - y_i \rvert \Big)^{1/d}$ | The geometric mean of the per-dimension gaps, the $p \to 0$ limit of the power-mean family. A shared coordinate makes the distance zero, so a selection that keeps every pair apart under this metric is spread in every coordinate projection as well as in the full space -- the pair distance behind *maximum projection designs* (Joseph, Gul & Ba, 2015). |
| `along_axis(axis)` | $d = \lvert x_{\text{axis}} - y_{\text{axis}} \rvert$ | The distance along one coordinate axis, every other coordinate ignored. A selection kept apart under it is spread along that single coordinate. |
| `l2_and_projections(l2_scale=1.0)` | $d = \min\Big( \min_i \lvert a_i - b_i \rvert,\; s \, \lVert a - b \rVert_2^{\,d} \Big)$ | For vectors $a$ and $b$ of dimension $d$, with $s$ = `l2_scale`: the smaller of the `l_minus_inf()` distance and the L2 part, $s$ times the L2 distance raised to the power $d$. Under `MIN_SEPARATION` a selection is spread in its projection onto every coordinate axis and in the full space at once. |

- **Speed depends on `p`.** The values $p \in \{1, 2, \infty, 0.5, 0.25, 0.125\}$ compute with hardware arithmetic; every other $p$ pays a `pow` call per dimension, well over an order of magnitude more per term.
- **`root=False` skips the outer $1/p$ root**, exactly as `l2s_euclidean_squared()` does for `l2_euclidean()` -- see that row above.
- **For $0 < p < 1$ the `root=True` form violates the triangle inequality** and is not a strict metric, while the `root=False` form is one -- the solver never relies on the triangle inequality, so both are usable.
- **`l2_and_projections()` assumes vectors scaled into the unit cube $[0,1]^d$**, the only population for which its 2 parts are comparable, and at least 2 dimensions: in 1 it only rescales the one coordinate gap.
- **In higher dimensions the L2 part rarely sets the minimum; an `l2_scale` below 1 lets the L2 part set the minimum more often.** An `l2_scale` of about 1/1.4 gives the 2 parts equal weight for $d = 3$, and about 1/40 for $d = 10$.
- **`geometric_mean()` is computed through logarithms**, one per dimension with a single exponential at the end, so the product cannot underflow or overflow at any dimension count.
- **Distinct points can be at distance zero under `geometric_mean()`, `l_minus_inf()`, `along_axis(axis)` and `l2_and_projections()`**; the solver treats them as a coincident pair -- see the [scoring page](scoring.md#diversity-tie-breakers) for the tie-breaker that removes one of them.

Reference for the geometric-mean distance: Joseph, V. R., Gul, E. & Ba, S. (2015). *Maximum projection designs for computer experiments*. Biometrika 102(2), 371–380. [doi:10.1093/biomet/asv002](https://doi.org/10.1093/biomet/asv002).

## III. Separation { #separation }

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

## IV. Mean distance { #mean-distance }

The **mean distance** of a selected item is its mean distance to the other selected items:

$$\text{md}(v) = \frac{1}{k - 1} \sum_{u \in S,\; u \neq v} d(v, u)$$

It is maintained incrementally as well, and even more cheaply than separation: adding an item
adds one distance to every item's running sum, and removing one subtracts it exactly -- no
rescan of the selection is ever needed.

## V. Diversity metrics { #diversity-metrics }

Each diversity metric aggregates the `k` separation values differently. (For how these objectives
map onto the operations-research literature -- p-dispersion, Max-SumMin, classical MaxSum MDP --
see [Objectives & the diversity-problem landscape](objectives.md).)

| Metric | Formula | Characteristics |
|--------|---------|-----------------|
| `GEOMEAN_SEPARATION` | $\exp\!\left(\frac{1}{k}\sum_{v \in S} \ln(\text{sep}(v))\right)$ | **Default.** Balances all separations. Sensitive to any item with low separation -- a single poorly-placed item drags down the score. |
| `MIN_SEPARATION` | $\min_{v \in S} \text{sep}(v)$ | Only considers the worst-off item (the closest pair). Equivalent to the *p-dispersion* problem. Many swaps produce tied scores. |
| `MEAN_SEPARATION` | $\frac{1}{k}\sum_{v \in S} \text{sep}(v)$ | Averages all separations. Less sensitive to individual outliers than geomean, but can be dominated by a few very high separations. |
| `APPROX_GEOMEAN_SEPARATION` | Same as geomean but using fast log/exp approximations | Slightly less accurate but faster per iteration. Useful for large-scale problems where iteration speed matters more than per-iteration precision. |
| `HARMONIC_MEAN_SEPARATION` | $k \,/\, \sum_{v \in S} \big(1 / \text{sep}(v)\big)$ | Between the geometric mean and the minimum: every item still counts, but a close pair lowers the score more than under the geomean. Zero as soon as one separation is zero. Computed exactly, without logarithm or exponential. |
| `MEAN_PAIRWISE_DISTANCE` | $\frac{2}{k(k-1)}\sum_{\{u,v\} \subseteq S} d(u, v)$ | Mean distance over all selected *pairs* -- the classical **max-sum diversity** objective (MaxSum MDP, also known as *remote-clique*). Maximizes total spread: selections gravitate to the outer regions of the data, and near-duplicates are tolerated if both sit far from everything else. |

### V.A. Which metric to choose? { #which-metric-to-choose }

- **`GEOMEAN_SEPARATION`** is the best default. It naturally penalizes any clustering in the
  selection while remaining smooth and differentiable in most of the search space.
- **`MIN_SEPARATION`** is appropriate when you specifically care about the worst-case nearest
  neighbor distance (e.g., facility placement where minimum coverage radius matters).
  Expect slower convergence due to many tied scores.
- **`MEAN_SEPARATION`** maximizes total spread. It may tolerate some clustering as long as
  other items compensate with large separations.
- **`APPROX_GEOMEAN_SEPARATION`** is a drop-in replacement for `GEOMEAN_SEPARATION` when
  you want to trade a small amount of precision for more iterations per second.
- **`HARMONIC_MEAN_SEPARATION`** sits between `GEOMEAN_SEPARATION` and `MIN_SEPARATION`: pick it
  when a close pair should weigh more than the geomean gives it, without the tied scores of the
  minimum.
- **`MEAN_PAIRWISE_DISTANCE`** is the objective to pick when you want classical max-sum
  diversity semantics ("maximize total spread") or want results comparable with the MaxSum
  MDP literature. Unlike every separation metric it does *not* penalize near-duplicates per
  se -- two nearly identical items at the data's boundary can both be kept. Note that the
  solver's swap heuristics were originally designed and tuned around separation contributions; they
  are correct for this metric (its per-item contribution is the item's exact marginal
  contribution to the objective), but the separation metrics remain the most battle-tested
  choice.
- **A [hybrid diversity metric](#hybrid-diversity-metrics)** when one notion of spread is not
  enough, such as spread in the full space and along each coordinate at once.

## VI. Hybrid diversity metrics { #hybrid-diversity-metrics }

A **hybrid diversity metric** combines several diversity metrics, its **terms**, into one score.

A hybrid is for selections that must be diverse in more than one sense at once: spread in the full space and along each coordinate, or spread under 2 different distances. The [uniform-sampling case study](../guides/uniform_sampling.md) compares a hybrid's selections with those of single diversity metrics.

### VI.A. Terms { #hybrid-terms }

A term is a diversity metric over one distance metric:

- `DiversityMetric.MIN_SEPARATION.over(DistanceMetric.along_axis(0))` reads the distance along axis 0;
- a bare `DiversityMetric.MIN_SEPARATION` reads the problem's own distance metric.

The distances under each term's distance metric are computed and stored in the same way as the distances under the problem's own distance metric, and every term adds work to each iteration. A hybrid needs at least 2 terms, and a hybrid cannot be a term of another hybrid.

A problem built from [precomputed distances](glossary.md#precomputed-distances) has no vectors, so none of its terms can use `.over(...)`; every term reads the given distances.

### VI.B. Aggregations { #hybrid-aggregations }

Each factory below combines the terms' values into the hybrid's score:

| Factory | Score | Choose it when |
|---------|-------|----------------|
| `HybridDiversityMetric.geomean_of(...)` | the geometric mean of the terms | **the selection must be spread under every term**: one term at zero makes the score zero |
| `HybridDiversityMetric.mean_of(...)` | the arithmetic mean of the terms | **a strong term may make up for a weak one** |
| `HybridDiversityMetric.min_of(...)` | the smallest of the terms | **the weakest term decides**: a high value in one term cannot make up for a low value in another |

```python
from max_div.metrics import DistanceMetric, DiversityMetric, HybridDiversityMetric

objective = HybridDiversityMetric.geomean_of(
    DiversityMetric.MIN_SEPARATION.over(DistanceMetric.l2_euclidean()),
    DiversityMetric.MIN_SEPARATION.over(DistanceMetric.along_axis(0)),
    DiversityMetric.MIN_SEPARATION.over(DistanceMetric.along_axis(1)),
)
problem = MaxDivProblem.new(vectors, k=100, diversity_metric=objective)
```

### VI.C. Weights { #hybrid-weights }

Every factory takes `weights=`: one positive, finite number per term, in term order; leaving out `weights=` gives every term weight 1. What a weight does depends on the aggregation, with $s_t$ the value of term $t$ and $w_t$ its weight:

| Factory | Score | A weight … |
|---------|-------|------------|
| `geomean_of` | $\Big( \prod_t s_t^{\,w_t} \Big)^{1 / \sum_t w_t}$ | is its term's exponent: a larger weight makes the score more sensitive to that term |
| `mean_of` | $\sum_t w_t \, s_t \,/\, \sum_t w_t$ | multiplies its term's value in a weighted mean |
| `min_of` | $\min_t \; w_t \, s_t$ | multiplies its term's value; the weights are not normalized |

- **The best selection under the geometric mean does not depend on the scale of a term**: multiplying a term by a constant multiplies every selection's score by the same factor, so the best selection stays the same.
- **The arithmetic mean and the minimum compare raw values**, so terms on different scales need weights that bring them onto a common scale.

For example, when the x coordinates span 0 to 1,000 and the y coordinates 0 to 1, the unweighted minimum of the x-axis term and the y-axis term almost always equals the y-axis term's value. Dividing each term by the range of its coordinate puts both on one scale:

```python
x_range = vectors[:, 0].max() - vectors[:, 0].min()
y_range = vectors[:, 1].max() - vectors[:, 1].min()

objective = HybridDiversityMetric.min_of(
    DiversityMetric.MIN_SEPARATION.over(DistanceMetric.along_axis(0)),
    DiversityMetric.MIN_SEPARATION.over(DistanceMetric.along_axis(1)),
    weights=(1 / x_range, 1 / y_range),
)
```

A `min_of` hybrid whose terms all use min-separation scores a selection exactly as min-separation over a single distance: the weighted minimum of the terms' distances.

[`l2_and_projections()`](#distance-metrics) is such a distance: the minimum of the `l_minus_inf()` distance and `l2_scale` times the L2 distance raised to the power $d$, the number of dimensions. Under min-separation it spreads a selection along every axis and in the full space at once.

### VI.D. Tie-breakers { #hybrid-tie-breakers }

A hybrid accepts no custom tie-breakers: `MaxDivSolverBuilder.with_diversity_tie_breakers()` raises a `ValueError` for a problem with a hybrid diversity metric. Its default tie-breakers come from the diversity metrics of its terms, and a `min_of` hybrid gets the unweighted geometric mean of its terms as its first tie-breaker. The [scoring page](scoring.md#diversity-tie-breakers) gives the full rule.
