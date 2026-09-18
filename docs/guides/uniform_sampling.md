# Case study: maximally uniform sampling in 2D and its marginals

!!! info "In short"
    A selection can be spread uniformly over the unit square and, at the same time, uniformly along each axis. Seven experiments on one population show what each [distance metric](../concepts/glossary.md#distance-metric) delivers on those three goals, that a [hybrid objective](../reference/metrics/HybridDiversityMetric.md) with one term per goal delivers all three at once, and what exact per-band counts cost on top of it.

## I. Problem statement

### I.A. Motivation

Some applications need $k$ points that look **uniformly spread over a hypercube without forming a regular grid**: the inputs of a test campaign, or the starting points of an optimizer. Often the **marginal distributions matter as well**: when each input dimension is tested on its own, the $x$ values alone should cover $[0, 1]$ evenly, and so should the $y$ values.

A selection that is uniform in the square is not automatically uniform in its marginals: two points can be far apart in 2D while sharing almost the same $x$ value.

### I.B. The three goals

Take the two-dimensional case. Given $n$ points drawn uniformly at random from the unit square, **select $k$ of them** such that

- **spread in the square:** every point is far from its nearest neighbor under the L2 distance;
- **spread along $x$:** every $x$ value is far from its nearest other $x$ value;
- **spread along $y$:** likewise.

The three goals compete for the same $k$ points. **How to trade them off is left open** here; the experiments show what each objective delivers on all three.

### I.C. How the experiments run

- **One population for every experiment:** $n = 10{,}000$ random points, from which $k = 100$ are selected, so the results are comparable.
- **One diversity metric for every experiment:** the [harmonic-mean separation](../concepts/diversity.md#diversity-metrics), the harmonic mean, over the selected points, of each point's distance to its nearest other point. It sits between the minimum and the geometric mean in how hard it penalizes one close pair; on a perfectly regular selection the minimum, harmonic-mean and geometric-mean separations all agree.
- **One solver setting for every experiment:** 16 workers within a 60 s end-to-end budget. One extra run, in section V.C, keeps everything else and lengthens the budget to 900 s.
- **One measure for every result:** the harmonic-mean separation of the selection under the L2, $x$ and $y$ distances, one per goal.

## II. Diversity references

Before optimizing, it helps to know what each goal could reach if it were the only one, with the $k$ points placed freely, not chosen from the population.

- **Along one axis.** $k$ values spread evenly over $[0, 1]$, the first at 0 and the last at 1, are $1 / (k - 1)$ apart, and no placement does better on the minimum. The reference for the $x$ and $y$ goals is $1 / (k - 1) = 1/99$.
- **In the square.** For $k = 100$ a $10 \times 10$ grid over $[0, 1]^2$ places the points $1 / (\sqrt{k} - 1) = 1/9 \approx 0.111$ apart. The densest known arrangement does 3 % better: the best known packing of 100 equal circles in a square, hexagonal in its interior, has radius $r = 0.051401$,[^packomania] and its centers, which lie in the inner square of side $1 - 2r$, are $2r / (1 - 2r) \approx 0.1146$ apart in the unit square. That spacing is the reference for the L2 goal.

[^packomania]: Specht, E. *Packomania*, the best known packings of equal circles in a square, [N = 97 to 108](https://www.packomania.com/csq/pdf/d9.pdf).

The references are for free placement; a selection from $n$ random points falls short of them, more so the smaller $n$ is. In every table below the achieved separation is given as a fraction of its reference.

## III. One goal at a time

Each experiment here maximizes one goal and ignores the other two: it shows how close the selection gets to the reference, and what the other two goals lose.

In every figure the red dots are the selection, with its $x$ and $y$ values as rug marks along the bottom and left edges. Hover over a dot to see:

- its nearest neighbor under the objective's distance in blue, and the level curve of that distance through the neighbor;
- its nearest neighbors under the three reference distances, each drawn as a dashed ring: a central dot marks the L2 neighbor, a vertical stroke the $x$ neighbor, a horizontal stroke the $y$ neighbor.

### III.A. L2 distance

--8<-- "generated/uniform_sampling_l2_figure.html"

--8<-- "generated/uniform_sampling_l2_separations.md"

The selection is spread in the square, but along either axis the values cluster and leave gaps, because the objective does not measure them. The level curve through a point's nearest neighbor is a circle, and the marginal neighbors sit far outside it.

### III.B. $x$ distance

--8<-- "generated/uniform_sampling_x_figure.html"

--8<-- "generated/uniform_sampling_x_separations.md"

The $x$ values reach almost even spacing, and the $y$ values are as random as the population. The objective measures only $x$, so every selection with the same $x$ values scores the same, and the solver has no reason to prefer one $y$ arrangement over another. The level curve is a pair of vertical lines: the objective distance between two points is the horizontal gap, whatever their vertical gap.

### III.C. $y$ distance

--8<-- "generated/uniform_sampling_y_figure.html"

--8<-- "generated/uniform_sampling_y_separations.md"

The mirror image of III.B: the $y$ values are evenly spread and the $x$ values are not.

## IV. One distance covering several goals

A single objective can still cover several goals when its distance combines them. Two distances do that.

### IV.A. L−∞ distance

The [L−∞ distance](../concepts/diversity.md#distance-metrics) between two points is the smaller of their two coordinate gaps, so two points are far apart under it only when they are far apart along $x$ *and* along $y$. Keeping every pair apart under it spreads the selection along both axes at once.

--8<-- "generated/uniform_sampling_linf_figure.html"

--8<-- "generated/uniform_sampling_linf_separations.md"

Both marginals come close to their references. The distance says nothing about the L2 spread: the level curve through a point's nearest neighbor traces the four half-lines where one coordinate gap equals $d$ and the other is larger, a square outline whose sides extend outward past the corners, and two points diagonally across the square can be at a small L−∞ distance. Nearby pairs in 2D are only kept apart as far as their coordinate gaps happen to require.

### IV.B. Geometric-mean distance

The [geometric-mean distance](../concepts/diversity.md#distance-metrics) is the square root of the product of the two coordinate gaps. It is a smoothed L−∞ distance: a small gap along one axis makes the distance small, but a large gap along the other axis compensates, in part.

Its level curves are hyperbolas $|\Delta x| \cdot |\Delta y| = d^2$. The curve at $d = 1/\sqrt{k}$ passes through two kinds of neighbor at once:

- one at the 2D spacing in both coordinates;
- one across the whole square in one coordinate and at the marginal spacing in the other.

The distance therefore covers all three goals in one number, as the maximum projection designs of Joseph, Gul & Ba (2015) do with the same product.[^maxpro]

--8<-- "generated/uniform_sampling_geomean_figure.html"

--8<-- "generated/uniform_sampling_geomean_separations.md"

The selection is spread in the square and along both axes, none of the three at its single-goal level. The nearest neighbor under this distance is often a point that shares almost the same $x$ or $y$ value, at a large gap in the other coordinate: the pair relevant to the marginal goal, weighted by how far apart it is in the other direction.

[^maxpro]: Joseph, V. R., Gul, E. & Ba, S. (2015). *Maximum projection designs for computer experiments*. Biometrika 102(2), 371–380. [doi:10.1093/biomet/asv002](https://doi.org/10.1093/biomet/asv002). Their criterion sums, over all pairs, the reciprocal of the product of the squared coordinate gaps, and places the points freely; the experiment here maximizes the nearest-neighbor separation of a selection from a fixed population.

## V. Hybrid objective

A hybrid objective states the three goals directly: one term per goal, each the harmonic-mean separation under that goal's distance, combined by their geometric mean so that no term dominates by its scale.

### V.A. Unconstrained

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

### V.B. Exact counts per band

The same objective, under [constraints](../concepts/constraints.md): the unit square is cut into five equal bands along $x$ and five along $y$, and each of the ten bands must hold exactly 20 of the 100 selected items. Each band is one constraint over the population items whose coordinate falls in it:

```python
import numpy as np

from max_div.problem import Constraint, MaxDivProblem

n_bands, k = 5, 100
band_indices = np.minimum((vectors * n_bands).astype(int), n_bands - 1)  # per item, per axis
constraints = [
    Constraint(
        int_set=set(np.flatnonzero(band_indices[:, axis] == band)),
        min_count=k // n_bands,
        max_count=k // n_bands,
    )
    for axis in (0, 1)
    for band in range(n_bands)
]
problem = MaxDivProblem.new(vectors=vectors, k=k, diversity_metric=objective, constraints=constraints)
```

The light gray lines are the band edges.

--8<-- "generated/uniform_sampling_hybrid_banded_figure.html"

--8<-- "generated/uniform_sampling_hybrid_banded_separations.md"

Every band holds its 20 items; the unconstrained selection of V.A holds between 17 and 22 per band. The three separations are the same as in V.A to within 1 % of their references: on this population the exact counts cost no diversity.

The exact counts make each iteration slower, since each candidate swap is also checked against the ten counts; the convergence table below shows the resulting lower iteration count.

### V.C. What a longer budget improves

Every figure above is a 60 s solve, chosen so the whole case study regenerates in minutes. This one is the same problem as V.B solved for 900 s, with the solver built with `with_intermediate_selections()` so every [score checkpoint](../concepts/parallel_solving.md#reading-the-result) also carries the selection held at that moment.

The figure steps through those selections: each frame is a checkpoint at which the best selection across the 16 workers changed, and the caption gives the frame's elapsed time and diversity. Drag the slider, use the buttons, or the arrow keys once the figure has focus.

--8<-- "generated/uniform_sampling_hybrid_banded_long_replay.html"

--8<-- "generated/uniform_sampling_hybrid_banded_long_separations.md"

Most frames sit in the first minute, where the selection still changes at nearly every checkpoint. After that a change is rare, and it is one of two kinds:

- a swap of two or three items, or
- a wholesale change, when another worker's selection overtakes the best held so far and becomes the new best-known selection.

The later frames are where the extra budget improves the result: the diversity keeps climbing past V.B's 60 s value, so the summary table below lists this run as its own row.

## VI. Summary

Every experiment's achieved harmonic-mean separation under the three reference distances, each as a fraction of its free-placement reference from section II. A result <span class="usx-low">below 50 %</span> of its reference is marked red, one <span class="usx-high">above 70 %</span> green:

--8<-- "generated/uniform_sampling_summary.md"

- **One distance reaches one goal.** The L2, $x$ and $y$ distances each reach their own goal and leave at least one other near the level of a random selection.
- **The L−∞ distance reaches the two marginals**, and the geometric-mean distance gets part of the way on all three.
- **The hybrid objective directly optimizes all three** by explicitly formulating the three objectives, at the cost of slower iterations due to the three objectives.
- **Exact counts per band come at no cost in diversity**: under them the hybrid objective reaches the same three separations.
- **A longer budget still improves the result**: the 900 s solve of V.C ends above its 60 s counterpart on the L2 and $x$ separations, and level on $y$.

The slower iterations are visible in the iteration counts. The table gives, per experiment, how many iterations the worker holding the final selection completed in the 60 s budget, and the best objective any worker held at three elapsed marks as a fraction of the final value:

--8<-- "generated/uniform_sampling_convergence.md"
