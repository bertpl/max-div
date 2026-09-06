# When to use the geometric-mean distance

!!! info "In short"
    This page shows a common use case where the geometric-mean [distance metric](../concepts/glossary.md#distance-metric) is particularly useful, beyond what might be self-evident from looking at the metric's formula itself.

## I. Randomized uniform selections use case

Some applications need a set of $k$ tuples that look uniformly spread over a hypercube, without being a regular grid: the inputs of a test campaign, the starting points of an optimizer, the settings of a parameter sweep. Take the two-dimensional case, $k$ tuples $(x_i, y_i)$ in the unit square:

$$S = \{(x_i, y_i) \mid i = 1, \ldots, k\}, \qquad 0 \le x_i, y_i \le 1.$$

Uniform spread in the square means every tuple is far from its nearest neighbor. With $k$ points sharing a unit of area, the room per point is $1/k$, so the ideal nearest-neighbor distance is about its square root:

$$\text{sep}_{2D}(i) = \min_{j \neq i} \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2} \;\approx\; \frac{1}{\sqrt{k}}.$$

Often the marginal distributions matter as well: when each input dimension is tested on its own, the $x$ values alone should also cover $[0, 1]$ evenly, and so should the $y$ values. Along one dimension the $k$ values share a unit of length, so the ideal nearest-neighbor distance is about $1/k$:

$$\begin{aligned}
\text{sep}_x(i) &= \min_{j \neq i} |x_i - x_j| \;\approx\; \frac{1}{k}, \\
\text{sep}_y(i) &= \min_{j \neq i} |y_i - y_j| \;\approx\; \frac{1}{k}.
\end{aligned}$$

A selection that is uniform in the square is not automatically uniform in its marginals: two points can be far apart in 2D while sharing the same $x$ value. Maximizing the usual Euclidean separation delivers the first property and ignores the second.

## II. Geometric-mean distance metric

The geometric-mean distance between two tuples is the geometric mean of their per-dimension gaps, in two dimensions

$$d(p, q) = \sqrt{|x_p - x_q| \cdot |y_p - y_q|},$$

and in $d$ dimensions the $d$-th root of the product of all gaps. A shared coordinate value makes the distance zero, whatever the other coordinates do.

Its level curves are hyperbolas $|\Delta x| \cdot |\Delta y| = d^2$. The curve at $d = 1/\sqrt{k}$ passes through both ideals of section I at once: the point $(1/\sqrt{k}, 1/\sqrt{k})$, where a neighbor sits at the 2D-uniform spacing in both coordinates, and the point $(1, 1/k)$, where a neighbor is as far away as the square allows in one coordinate and at the marginal-uniform spacing in the other. The figure shows the curve family for $k = 25$:

![Level curves of the geometric-mean distance from the origin in the unit square, with the curve at 1/√k passing through (1/√k, 1/√k) and (1, 1/k)](./images/geomean_distance_levels.webp)

So a selection whose smallest geometric-mean distance is about $1/\sqrt{k}$ keeps every pair of points either apart in the 2D sense or apart in each marginal, and the closer $k$ gets to the sizes where the $1/k$ and $1/\sqrt{k}$ spacings hold exactly, the tighter the match. Maximizing separation under this metric therefore spreads the selection in the square and in its marginals at the same time.

## III. Example

A population of $n = 50{,}000$ points sampled uniformly and independently per coordinate in the unit square, from which $k = 100$ are selected under geometric-mean separation with the geometric-mean distance, on 16 workers with a 60 s end-to-end budget:

![Fifty thousand grey points in the unit square with the hundred selected ones in red, and the selection's x and y values as rug marks along the bottom and left edges](./images/geomean_distance_example.webp)

The red points spread over the square, and the rug marks along the two edges show the marginals: the hundred $x$ values and the hundred $y$ values each cover $[0, 1]$ without gaps or clusters.
