# Case study: maximally uniform sampling in 2D and its marginals

!!! info "In short"
    A selection can be spread uniformly over the unit square and, at the same time, uniformly along each axis. Six experiments on one population show what each [distance metric](../concepts/glossary.md#distance-metric) delivers on those three goals, and that a [hybrid objective](../reference/metrics/HybridDiversityMetric.md) with one term per goal delivers all three at once.

## I. Problem statement

Some applications need $k$ points that look uniformly spread over a hypercube without forming a regular grid: the inputs of a test campaign, or the starting points of an optimizer. Often the marginal distributions matter as well: when each input dimension is tested on its own, the $x$ values alone should cover $[0, 1]$ evenly, and so should the $y$ values.

Take the two-dimensional case. Given $n$ points drawn uniformly at random from the unit square, select $k$ of them such that

- the selection is spread in the square: every point is far from its nearest neighbor under the L2 distance;
- the selection is spread along $x$: every $x$ value is far from its nearest other $x$ value;
- the selection is spread along $y$: likewise.

The three goals compete for the same $k$ points, and this page leaves open how to trade them off. A selection that is uniform in the square is not automatically uniform in its marginals: two points can be far apart in 2D while sharing almost the same $x$ value.

Every experiment below selects $k = 100$ points from the same population of $n = 10{,}000$, so the results compare. A goal is measured as the [harmonic-mean separation](../concepts/diversity.md#diversity-metrics) under its distance: the harmonic mean over the selection of each point's distance to its nearest other point. It sits between the minimum and the geometric mean in how hard it penalizes one close pair; on a perfectly regular selection, where every nearest-neighbor distance is the same, all three aggregates agree. Every run maximizes that harmonic-mean separation on 16 workers within a 60 s end-to-end budget, and every result table reports it under the L2, $x$ and $y$ distances.

## II. Diversity references

Before optimizing, it helps to know what each goal could reach if it were the only one, with the $k$ points placed freely instead of chosen from the population.

- **Along one axis.** $k$ values spread evenly over $[0, 1]$, the first at 0 and the last at 1, are $1 / (k - 1)$ apart, and no placement does better on the minimum. The reference for the $x$ and $y$ goals is $1 / (k - 1) = 1/99$.
- **In the square.** For $k = 100$ a $10 \times 10$ grid over $[0, 1]^2$ places the points $1 / (\sqrt{k} - 1) = 1/9 \approx 0.111$ apart. The densest known arrangement does 3 % better: the best known packing of 100 equal circles in a square, hexagonal in the bulk, has radius $r = 0.051401$,[^packomania] and its centers, which lie in the inner square of side $1 - 2r$, are $2r / (1 - 2r) \approx 0.1146$ apart in the unit square. That spacing is the reference for the L2 goal.

[^packomania]: Specht, E. *Packomania*, the best known packings of equal circles in a square, [N = 97 to 108](https://www.packomania.com/csq/pdf/d9.pdf).

The references are for free placement; a selection from $n$ random points falls short of them, more so the smaller $n$ is. In every table below the achieved separation is given as a fraction of its reference.

## III. One goal at a time

Each experiment here maximizes one goal and ignores the other two: it shows how close the selection gets to the reference, and what the other two goals lose.

In every figure the red dots are the selection, with its $x$ and $y$ values as rug marks along the bottom and left edges. Hover over a dot to see its nearest neighbor under the objective's distance in blue, the level curve of that distance through the neighbor, and its nearest neighbors under the three reference distances: a dashed ring with a central dot for L2, with a vertical stroke for $x$, with a horizontal stroke for $y$.

### III.A. L2 distance

--8<-- "generated/uniform_sampling_l2_figure.html"

--8<-- "generated/uniform_sampling_l2_separations.md"

The selection is spread in the square, and the rug marks show the price: along either axis the values cluster and leave gaps, because nothing in the objective sees them. The level curve through a point's nearest neighbor is a circle, and the marginal neighbors sit far outside it.

### III.B. $x$ distance

--8<-- "generated/uniform_sampling_x_figure.html"

--8<-- "generated/uniform_sampling_x_separations.md"

The $x$ values reach almost even spacing, and the $y$ values are as random as the population. The objective sees only $x$, so every selection with the same $x$ values scores the same, and the solver has no reason to prefer one $y$ arrangement over another. The level curve is a pair of vertical lines: the objective distance between two points is the horizontal gap, whatever their vertical gap.

### III.C. $y$ distance

--8<-- "generated/uniform_sampling_y_figure.html"

--8<-- "generated/uniform_sampling_y_separations.md"

The mirror image of III.B: the $y$ values are evenly spread and the $x$ values are not.

## IV. One distance covering several goals

A single objective can still see several goals when its distance combines them. Two distances do that.

### IV.A. L−∞ distance

The [L−∞ distance](../concepts/diversity.md#distance-metrics) between two points is the smaller of their two coordinate gaps, so two points are far apart under it only when they are far apart along $x$ *and* along $y$. Keeping every pair apart under it spreads the selection along both axes at once.

--8<-- "generated/uniform_sampling_linf_figure.html"

--8<-- "generated/uniform_sampling_linf_separations.md"

Both marginals come close to their references. The distance says nothing about the L2 spread: the level curve through a point's nearest neighbor is the edge of a square around the point, extended outward, and two points diagonally across the square can be at a small L−∞ distance. Nearby pairs in 2D are only kept apart as far as their coordinate gaps happen to require.

### IV.B. Geometric-mean distance

The [geometric-mean distance](../concepts/diversity.md#distance-metrics) is the square root of the product of the two coordinate gaps. It is a smoothed L−∞ distance: a small gap along one axis makes the distance small, but a large gap along the other axis compensates, in part. Its level curves are hyperbolas $|\Delta x| \cdot |\Delta y| = d^2$, and the curve at $d = 1/\sqrt{k}$ passes through a neighbor at the 2D spacing in both coordinates and through a neighbor across the whole square in one coordinate and at the marginal spacing in the other. The distance therefore covers all three goals in one number, as the maximum projection designs of Joseph, Gul & Ba (2015) do with the same product.[^maxpro]

--8<-- "generated/uniform_sampling_geomean_figure.html"

--8<-- "generated/uniform_sampling_geomean_separations.md"

The selection is spread in the square and along both axes, none of the three at its single-goal level. The nearest neighbor under this distance is often a point that shares almost the same $x$ or $y$ value, at a large gap in the other coordinate: the pair the marginal goal cares about, weighted by how far apart it is in the other direction.

[^maxpro]: Joseph, V. R., Gul, E. & Ba, S. (2015). *Maximum projection designs for computer experiments*. Biometrika 102(2), 371–380. [doi:10.1093/biomet/asv002](https://doi.org/10.1093/biomet/asv002). Their criterion sums, over all pairs, the reciprocal of the product of the squared coordinate gaps, and places the points freely; the experiment here maximizes the nearest-neighbor separation of a selection from a fixed population.

## V. Hybrid objective

A hybrid objective states the three goals directly: one term per goal, each the harmonic-mean separation under that goal's distance, combined by their geometric mean so that no term dominates by its scale.

```python
from max_div.metrics import DistanceMetric, DiversityMetric, HybridDiversityMetric

objective = HybridDiversityMetric.geomean_of(
    DiversityMetric.HARMONIC_MEAN_SEPARATION.over(DistanceMetric.l2_euclidean()),
    DiversityMetric.HARMONIC_MEAN_SEPARATION.over(DistanceMetric.along_axis(0)),
    DiversityMetric.HARMONIC_MEAN_SEPARATION.over(DistanceMetric.along_axis(1)),
)
```

Hover over a dot to see the three level curves, each through the point's nearest neighbor under its term's distance.

--8<-- "generated/uniform_sampling_hybrid_figure.html"

--8<-- "generated/uniform_sampling_hybrid_separations.md"

The three goals are each close to their reference, at the same time.

## VI. Summary

Every experiment's achieved harmonic-mean separation under the three reference distances, each as a fraction of its free-placement reference from section II:

--8<-- "generated/uniform_sampling_summary.md"

- **One distance serves one goal.** The L2, $x$ and $y$ distances each reach their own goal and leave at least one other near the level of a random selection.
- **The L−∞ distance serves the two marginals**, and the geometric-mean distance serves all three at a discount on each.
- **The hybrid objective serves all three** by naming them: it is the choice when every goal is a requirement, at the cost of tracking three distances instead of one.

That cost shows in the iterations each run fits into its budget. The table gives them for the winning worker, and the objective it had reached at three elapsed marks as a fraction of its final value:

--8<-- "generated/uniform_sampling_convergence.md"
