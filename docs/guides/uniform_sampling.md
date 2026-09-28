# Case study: maximally uniform sampling in 2D and its marginals

!!! info "In short"
    A selection can be spread uniformly over the unit square and, at the same time, uniformly along each axis. The guide's experiments on one population show:

    - what each [distance metric](../concepts/glossary.md#distance-metric) delivers on those 3 goals;
    - that a [hybrid objective](../reference/metrics/HybridDiversityMetric.md) with one term per goal delivers all 3 at once;
    - what exact per-band counts cost on top of it.

    A hybrid that takes the minimum over weighted terms, not their geometric mean, spreads the selection further over the square without spreading it less along either axis. Min separation under the marginals-and-joint distance, a single distance that is the minimum of an L−∞ part and an L2 part, gives the lowest of the 3 goals the highest value of any 60 s experiment.

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
- **One diversity metric for every experiment:** the [min separation](../concepts/diversity.md#diversity-metrics), the smallest distance from any selected point to its nearest other selected point. It is the strictest of the separation metrics: one close pair sets the score, whatever the rest of the selection looks like.
- **Ties are broken by the solver's default rule:** many selections share the same closest pair, so the solver's default [tie-breakers](../concepts/scoring.md#diversity-tie-breakers) decide between them.
- **One solver setting for every experiment:** 16 workers within a 60 s end-to-end budget. 2 extra runs, in section V.C, solve the problems of V.A and V.B again with 32 workers for 4 h.
- **One measure for every result:** the min separation of the selection under the L2, $x$ and $y$ distances, one per goal.

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

A single objective can still cover several goals when its distance combines them. The 2 distances here do that only in part: the L−∞ distance of IV.A covers the 2 marginal goals and ignores the L2 goal, and the geometric-mean distance of IV.B approximates all 3 at once.

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

Each objective below combines a part for the L2 goal with parts for the marginal goals, and they differ in how they combine them:

- **V.A to V.C** use a [hybrid objective](../concepts/diversity.md#hybrid-diversity-metrics) with one term per goal, each the min separation under that goal's distance, combined by their geometric mean so that no term dominates by its scale.
- **V.D** takes the minimum of 2 weighted terms, so the lowest weighted term sets the score.
- **V.E** uses a single distance, the marginals-and-joint distance, that is the minimum of an L−∞ part and an L2 part, with no weights.

### V.A. Unconstrained

```python
from max_div.metrics import DistanceMetric, DiversityMetric, HybridDiversityMetric

objective = HybridDiversityMetric.geomean_of(
    DiversityMetric.MIN_SEPARATION.over(DistanceMetric.l2_euclidean()),
    DiversityMetric.MIN_SEPARATION.over(DistanceMetric.along_axis(0)),
    DiversityMetric.MIN_SEPARATION.over(DistanceMetric.along_axis(1)),
)
```

Hover over a dot to see the three level curves, each through the point's nearest neighbor under its term's distance.

--8<-- "generated/uniform_sampling_hybrid_geomean_figure.html"

--8<-- "generated/uniform_sampling_hybrid_geomean_separations.md"

The two marginal goals reach about 70 % of their references and the L2 goal reaches more than half of its reference, all three at the same time; no single-distance experiment comes close on the two goals it ignores.

### V.B. Banded constraints

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

--8<-- "generated/uniform_sampling_hybrid_geomean_banded_figure.html"

--8<-- "generated/uniform_sampling_hybrid_geomean_banded_separations.md"

Every band holds its 20 items; the unconstrained selection of V.A holds between 16 and 22 per band. The three separations stay within 3 % of V.A's separations: on this population the exact counts cost almost no diversity.

The exact counts make each iteration slower, since each candidate swap is also checked against the ten counts; the convergence table below shows the resulting lower iteration count.

### V.C. Long-budget runs

Every figure above is a 60 s solve, chosen so the whole case study regenerates in minutes. This section solves the 2 hybrid problems of V.A and V.B again with 32 workers for 4 h, on a 16-core machine, so 2 workers share each core.

The solver is built with `with_intermediate_selections()` so every [score checkpoint](../concepts/parallel_solving.md#reading-the-result) also carries the selection held at that moment.

The 2 figures step through those selections: each frame is a checkpoint at which the best selection across the 32 workers changed, and the caption gives the frame's elapsed time and diversity. Move between frames 3 ways:

- drag the slider,
- click the buttons, or
- press the arrow keys once the figure has focus.

#### V.C.1. Unconstrained

--8<-- "generated/uniform_sampling_hybrid_geomean_long_replay.html"

The [timeline of this solve](images/uniform_sampling_hybrid_geomean_long_timeline.webp) shows:

- the 32 workers,
- their groups merging over the 4 h, and
- the trajectories of the objective and its [tie-breakers](../concepts/scoring.md#diversity-tie-breakers).

The [parallel-solving page](../concepts/parallel_solving.md#the-three-groupings-on-one-problem) explains how to read it.

--8<-- "generated/uniform_sampling_hybrid_geomean_long_separations.md"

Most frames fall in the first minute, where the selection still changes at nearly every checkpoint. After that a change is rare, and it is one of two kinds:

- a change of between 2 and about 20 items, or
- a wholesale change, when another worker's selection surpasses the best held so far and becomes the new best-known selection.

The later frames are where the extra budget improves the result: the diversity ends 5.5 % above V.A's 60 s value, and it was still increasing in the last hour, with its last 2 improvements after 3 h 20 m.

#### V.C.2. Banded constraints

The [timeline of this solve](images/uniform_sampling_hybrid_geomean_banded_long_timeline.webp) also shows the constraints score.

--8<-- "generated/uniform_sampling_hybrid_geomean_banded_long_replay.html"

--8<-- "generated/uniform_sampling_hybrid_geomean_banded_long_separations.md"

The diversity ends 7.7 % above V.B's 60 s value, with its last improvement at about 3 h 20 m.

**Under this budget the banded solve ends ahead of the unconstrained one on the hybrid objective**, 0.01583 against 0.01573, where at 60 s the banded solve was behind, 0.01470 against 0.01491.

Every selection that meets the band counts is also a valid unconstrained selection, so the unconstrained problem has a selection at least as good as the banded result, and the unconstrained solve did not find it.

A likely reason is that the band counts shrink the search space, so the same 4 h of swaps cover a larger share of the selections that remain. Each problem was solved once, with 1 seed, and a gap of 0.6 % is within what a different seed can change, so the banded solve's lead is a hint, not an established result.

### V.D. Minimum of weighted terms

A [minimum](../concepts/diversity.md#hybrid-aggregations) makes the score equal to the lowest weighted term: a swap improves the score only by improving the term that is lowest. The objective covers the 3 goals with 2 terms: the L−∞ distance of IV.A covers both marginal goals, and the L2 distance covers the third.

A minimum compares its terms' values directly, so the terms need a common scale first.

Among $k$ points well spread in the square, the nearest-neighbor L−∞ distance is about $1/k$ and the L2 distance about $1/\sqrt{k}$. [Weights](../concepts/diversity.md#hybrid-weights) of $k$ and $\sqrt{k}$ multiply the L−∞ and L2 terms respectively, which brings both to about 1:

```python
import math

from max_div.metrics import DistanceMetric, DiversityMetric, HybridDiversityMetric

k = 100
objective = HybridDiversityMetric.min_of(
    DiversityMetric.MIN_SEPARATION.over(DistanceMetric.l_minus_inf()),
    DiversityMetric.MIN_SEPARATION.over(DistanceMetric.l2_euclidean()),
    weights=(k, math.sqrt(k)),
)
```

--8<-- "generated/uniform_sampling_hybrid_weighted_min_figure.html"

--8<-- "generated/uniform_sampling_hybrid_weighted_min_separations.md"

Against the geometric-mean hybrid of V.A, the L2 goal rises from 56 % to 63 % of its reference, and the 2 marginal goals stay at 72 % and 71 %. On each goal, this 60 s solve comes within 2 percentage points of V.C.1, which solved V.A's geometric-mean objective for 4 h.

With 2 terms instead of 3, each iteration is also faster, as the convergence table in VI shows.

### V.E. Marginals-and-joint distance

Min separation takes a minimum over pairs, and V.D's objective takes a minimum over terms. The 2 minimums can be swapped: taking, for each term, the weighted distance of its closest pair and then the smallest of those values gives the same number as taking, for each pair, the smallest of its weighted term distances and then the smallest over all pairs.

V.D's objective is therefore the min separation under a single distance, $\min(k \cdot d_{\text{L}-\infty},\ \sqrt{k} \cdot d_{\text{L2}})$.

The [marginals-and-joint distance](../concepts/diversity.md#distance-metrics) is, like V.D's single distance, a minimum of an L−∞ part and an L2 part, but with no weight that depends on $k$. In 2D it is

$$
d(a, b) = \min\Big( \min\big(\lvert a_x - b_x \rvert, \lvert a_y - b_y \rvert\big),\; \lVert a - b \rVert_2^{\,2} \Big)
$$

- **The first part is the L−∞ distance** of IV.A, which covers both marginal goals.
- **The second part is the L2 distance raised to the power of the dimension**, here squared. Among $k$ well-spread points in the square, both parts of a nearest-neighbor pair are about $1/k$, so neither part needs a weight that depends on $k$.

```python
from max_div.metrics import DistanceMetric, DiversityMetric
from max_div.problem import MaxDivProblem

problem = MaxDivProblem.new(
    vectors=vectors,
    k=100,
    distance_metric=DistanceMetric.marginals_and_joint(),
    diversity_metric=DiversityMetric.MIN_SEPARATION,
)
```

Hover over a dot to see the level curve of this distance: the L−∞ curve of IV.A, with the corner of each quadrant cut off by a circle. Points are close under the marginals-and-joint distance when they are close along one axis, or close under the L2 distance.

--8<-- "generated/uniform_sampling_marginals_and_joint_figure.html"

--8<-- "generated/uniform_sampling_marginals_and_joint_separations.md"

Against V.D, the L2 goal rises from 63 % to 72 % of its reference, and the 2 marginal goals drop from 72 % and 71 % to 66 % and 67 %. The lowest of the 3 goals, at 66 %, is the highest of any 60 s experiment.

A solve under a single distance also iterates as fast as the single-distance experiments of III and IV, as the convergence table in VI shows.

The L2 goal gains because V.D's weighted minimum and the marginals-and-joint distance turn the same L−∞ separation into different L2 separations. A solve that maximizes a minimum of 2 parts ends with the 2 parts about equal, so equating the 2 parts at each solve's achieved L−∞ separation gives the L2 separation that the solve should reach:

- **V.D:** $\sqrt{k} \cdot d_{\text{L2}} = k \cdot d_{\text{L}-\infty}$ gives $d_{\text{L2}} = \sqrt{k} \cdot d_{\text{L}-\infty} = 10 \times 0.0072 = 0.072$;
- **V.E:** $d_{\text{L2}}^{\,2} = d_{\text{L}-\infty}$ gives $d_{\text{L2}} = \sqrt{0.0067} = 0.082$.

Both match the L2 separations that the solves reached. `marginals_and_joint(joint_scale=...)` multiplies the second part by `joint_scale`: a value above 1 favors the 2 marginal goals, and one below 1 favors the L2 goal; this guide keeps the default of 1.

## VI. Summary

Every experiment's achieved min separation under the three reference distances, each as a fraction of its free-placement reference from section II. A result <span class="usx-low">at or below 40 %</span> of its reference is marked red, one <span class="usx-high">at or above 60 %</span> green:

--8<-- "generated/uniform_sampling_summary.md"

- **One distance reaches one goal.** The L2, $x$ and $y$ distances each reach their own goal and leave at least one other near the level of a random selection.
- **The L−∞ distance reaches the two marginals**, and the geometric-mean distance gets part of the way on all three.
- **The geometric-mean hybrid directly optimizes all 3** by explicitly formulating the 3 objectives, at the cost of slower iterations due to the 3 objectives.
- **Exact counts per band come at no cost in diversity**: under them the geometric-mean hybrid reaches the same 3 separations.
- **A longer budget still improves the result**: with 32 workers for 4 h, both 4 h solves end above their 60 s counterparts on all 3 separations, and both were still improving in the last hour.
- **A minimum of 2 weighted terms improves on the geometric mean of 3**: in 60 s it comes within 2 percentage points, on each goal, of what V.C.1 reaches on V.A's objective in 4 h.
- **The marginals-and-joint distance gives the lowest of the 3 goals the highest value of any 60 s experiment**, 66 % of its reference, with a single distance and no weights to choose.

The slower iterations are visible in the iteration counts. The table gives, per experiment, how many iterations the worker holding the final selection completed in the 60 s budget, and the best objective any worker held at three elapsed marks as a fraction of the final value:

--8<-- "generated/uniform_sampling_convergence.md"
