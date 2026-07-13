# Diversity & Distance

## From Vectors to Diversity

The solver evaluates how diverse a selection is through a three-step process:

```
Vectors  ──>  Pairwise Distances  ──>  Separations  ──>  Diversity Score
 (n x d)        (n*(n-1)/2)              (k x 1)           (scalar)
```

1. **Pairwise distances** are computed once upfront between all `n` vectors using the chosen
   distance metric. These are stored as a condensed distance vector (like scipy's `pdist`).

2. **Separations** are computed for each selected vector: the distance to its *nearest neighbor*
   within the current selection. This is the key quantity -- a vector with high separation is
   well-spread from the rest of the selection; a vector with low separation is close to at
   least one other selected vector.

3. The **diversity score** aggregates the `k` separation values into a single scalar using the
   chosen diversity metric.

## Distance Metrics

The distance metric determines how the distance between two vectors is measured.

| Metric | Formula | Notes |
|--------|---------|-------|
| `L2_EUCLIDEAN` | $$d = \sqrt{\sum_i (x_i - y_i)^2}$$ | Standard Euclidean distance. Default. |
| `L1_MANHATTAN` | $$d = \sum_i \lvert x_i - y_i \rvert$$ | Less sensitive to outlier dimensions. |
| `L2S_EUCLIDEAN_SQUARED` | $$d = \sum_i (x_i - y_i)^2$$ | Avoids the square root. Produces identical solutions to `L2_EUCLIDEAN` when used with `GEOMEAN_SEPARATION`, since the geometric mean preserves distance ordering regardless of squaring. |
| `COSINE` | $$d = 1 - \frac{x \cdot y}{\lVert x \rVert \, \lVert y \rVert}$$ | Angular distance in $[0, 2]$, invariant to vector magnitude -- the natural choice for embedding-style vectors. Undefined for zero vectors, which are rejected at problem construction. |

## Separation

The **separation** of a selected vector is its minimum distance to any other selected vector:

$$\text{sep}(v) = \min_{u \in S,\; u \neq v} \; d(v, u)$$

where $S$ is the current selection and $d$ is the chosen distance metric.

Separation is maintained incrementally during optimization:

- **Adding** a vector only requires checking its distances to all other vectors and
  updating any separations that decrease.
- **Removing** a vector requires recomputing separation only for vectors whose nearest
  neighbor was the removed vector.

This incremental update is much cheaper than recomputing all separations from scratch after
each swap.

## Diversity Metrics

Each diversity metric aggregates the `k` separation values differently:

| Metric | Formula | Characteristics |
|--------|---------|-----------------|
| `GEOMEAN_SEPARATION` | $$\exp\!\left(\frac{1}{k}\sum_{v \in S} \ln(\text{sep}(v))\right)$$ | **Default.** Balances all separations. Sensitive to any vector with low separation -- a single poorly-placed vector drags down the score. |
| `MIN_SEPARATION` | $$\min_{v \in S} \text{sep}(v)$$ | Only considers the worst-off vector (the closest pair). Equivalent to the *p-dispersion* problem. Many swaps produce tied scores. |
| `MEAN_SEPARATION` | $$\frac{1}{k}\sum_{v \in S} \text{sep}(v)$$ | Averages all separations. Less sensitive to individual outliers than geomean, but can be dominated by a few very high separations. |
| `APPROX_GEOMEAN_SEPARATION` | Same as geomean but using fast log/exp approximations | Slightly less accurate but faster per iteration. Useful for large-scale problems where iteration speed matters more than per-iteration precision. |

### Which metric to choose?

- **`GEOMEAN_SEPARATION`** is the best default. It naturally penalizes any clustering in the
  selection while remaining smooth and differentiable in most of the search space.
- **`MIN_SEPARATION`** is appropriate when you specifically care about the worst-case nearest
  neighbor distance (e.g., facility placement where minimum coverage radius matters).
  Expect slower convergence due to many tied scores.
- **`MEAN_SEPARATION`** maximizes total spread. It may tolerate some clustering as long as
  other vectors compensate with large separations.
- **`APPROX_GEOMEAN_SEPARATION`** is a drop-in replacement for `GEOMEAN_SEPARATION` when
  you want to trade a small amount of precision for more iterations per second.
