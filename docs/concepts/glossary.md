# Glossary

Terms used across the documentation, defined once. Each entry carries a stable anchor for linking.

The intended reader is a developer comfortable with basic mathematics but new to dispersion
problems — so each entry says what the term means *and* why it matters here, rather than giving a
one-line dictionary gloss.

[Anytime algorithm](#anytime-algorithm) ·
[Condensed distance matrix](#condensed-distance-matrix) ·
[Constraint group](#constraint-group) ·
[Constraint weight](#constraint-weight) ·
[Dispersion](#dispersion) ·
[Distance metric](#distance-metric) ·
[Diversity contribution](#diversity-contribution) ·
[Diversity metric](#diversity-metric) ·
[Feasibility](#feasibility) ·
[Geometric-mean objective](#geometric-mean-objective) ·
[Guarantee type](#guarantee-type) ·
[Item](#item) ·
[Local search](#local-search) ·
[Max-min objective](#max-min-objective) ·
[Max-sum objective](#max-sum-objective) ·
[Overlapping constraints](#overlapping-constraints) ·
[Penalty shape](#penalty-shape) ·
[Preset](#preset) ·
[Score components](#score-components) ·
[Selection](#selection) ·
[Separation](#separation) ·
[Solver step](#solver-step) ·
[Swap](#swap) ·
[Vector](#vector)

## Anytime algorithm { #anytime-algorithm }

An algorithm that holds a valid answer at all times and keeps improving it for as long as it is
allowed to run, so it can be stopped at any moment and still return something usable. max-div is
anytime by design: you give it a time budget or an iteration count, and it returns the best
[selection](#selection) found within it. This is the property that distinguishes it from
single-shot pickers, which produce one answer in one pass and cannot use a larger budget — see
[Comparison with Other Tools](../comparison.md).

## Condensed distance matrix { #condensed-distance-matrix }

The pairwise distances between all *n* items, stored as a flat array of the *n*(*n*−1)/2 unique
values rather than a full *n*×*n* square — the layout `scipy.spatial.distance.pdist` returns.
max-div computes this once at problem construction and reads from it throughout solving, which is
what makes each [swap](#swap) cheap. It is also the memory bottleneck: the array grows with the
square of *n*, so problem size is limited by memory long before it is limited by solving time. A
problem can be constructed directly from such a matrix instead of from
[vectors](#vector).

## Constraint group { #constraint-group }

A set of [item](#item) indices, together with a minimum and/or maximum number of them that the
[selection](#selection) must contain. This is how fairness requirements are expressed: "at least 3
and at most 7 from this group". A group is defined purely by *which items belong to it* and never
by their coordinates, so constraints apply identically whether the problem was built from vectors
or from a [condensed distance matrix](#condensed-distance-matrix). See
[Constraints](constraints.md).

## Constraint weight { #constraint-weight }

A per-group multiplier, 1 by default, scaling how much that group's violations move the
constraint component of the [score](#score-components). Raising one group's weight makes the
solver prioritize satisfying it when it cannot satisfy everything. Weights matter only in relation
to each other — scaling all of them changes nothing.

## Dispersion { #dispersion }

The operations-research term for the family of problems max-div solves: choose a subset that is
*spread out* under some measure of mutual distance. The literature on dispersion is where the
objectives below come from, and using its vocabulary is what makes results comparable with
published work. Note that "Maximum Diversity Problem" conventionally names one *specific*
dispersion objective — the [max-sum](#max-sum-objective) one — so max-div's name should be read in
the broad sense. [Objectives & the Diversity-Problem Landscape](objectives.md) maps each metric to
its literature family.

## Distance metric { #distance-metric }

The rule for measuring how far apart two items are: Euclidean, squared Euclidean, Manhattan, or
cosine. It is chosen once per problem and determines every distance in the
[condensed distance matrix](#condensed-distance-matrix). The choice interacts with the
[diversity metric](#diversity-metric): squared Euclidean and plain Euclidean, for instance, yield
identical selections under the [geometric-mean objective](#geometric-mean-objective), because
squaring preserves the ordering of distances. See [Diversity & Distance](diversity.md).

## Diversity contribution { #diversity-contribution }

A per-item number saying how much *that* item contributes to the current selection's diversity,
with higher meaning more. Which quantity it is depends on the objective's family: for the
separation-based objectives it is the item's [separation](#separation), and for the
[max-sum objective](#max-sum-objective) it is the item's mean distance to the other selected
items. Contributions are the intermediate layer between raw distances and the single diversity
number, and they are what the solver's heuristics read when deciding which items are worth
removing.

## Diversity metric { #diversity-metric }

The function that aggregates the *k* [diversity contributions](#diversity-contribution) into the
one scalar being maximized — a minimum, a mean, or a geometric mean. Choosing a diversity metric
*is* choosing the objective, and it changes what a good answer looks like rather than merely how
hard the solver works: see [max-min](#max-min-objective), [max-sum](#max-sum-objective), and the
[geometric-mean objective](#geometric-mean-objective).

## Feasibility { #feasibility }

Whether a [selection](#selection) satisfies every [constraint group](#constraint-group). Because
the [score components](#score-components) put constraints strictly above diversity, the solver
always moves toward feasibility first and optimizes diversity only among equally feasible
selections. A problem whose constraints cannot all be met does not fail: it degrades to the
least-violating selection the solver can find, which is often the useful answer in practice.

## Geometric-mean objective { #geometric-mean-objective }

Maximize the *geometric mean* of the selected items' [separations](#separation) — max-div's
default, and the objective for which we have found no established name or other implementation in
the dispersion literature. Two readings help: it is the [max-sum](#max-sum-objective) of
*log*-distances, and it is the **Nash social welfare** rule from fair-division theory applied to
spread. That makes it the principled middle ground between
[max-min](#max-min-objective), where only the worst-off item counts, and a plain mean, where only
the total counts. Every item's separation matters, one near-duplicate collapses the score, and no
single large separation can buy that back.

## Guarantee type { #guarantee-type }

What a tool promises about the quality of its answer, which is the only fair basis for comparing
tools of different kinds:

- **Proven optimum** — the answer is optimal and the solver can prove it, typically only at small
  problem sizes.
- **Approximation** — the answer is provably within a stated factor of optimal; greedy
  farthest-point construction is within a factor 2 for [max-min](#max-min-objective).
- **Heuristic** — no bound, but good answers in practice; max-div is here.
- **Sampler** — draws diverse subsets from a distribution rather than maximizing anything, so
  "quality" does not mean the same thing.

## Item { #item }

One of the *n* candidates the solver chooses from — the unit that gets selected, that
[constraint groups](#constraint-group) range over, and that a [selection](#selection) is made of.
*Item* is deliberately geometry-free: a problem may be defined by [vectors](#vector) or by a
[condensed distance matrix](#condensed-distance-matrix) alone, and in the latter case items have
no coordinates at all.

## Local search { #local-search }

The optimization approach max-div uses: start from some [selection](#selection) and repeatedly
make small modifications, keeping those that improve the [score](#score-components). It offers no
guarantee of finding the best possible selection — the [guarantee type](#guarantee-type) is
*heuristic* — but it improves steadily with the budget it is given, which is what makes the solver
[anytime](#anytime-algorithm). Each modification is a [swap](#swap).

## Max-min objective { #max-min-objective }

Maximize the smallest [separation](#separation) in the selection — equivalently, push the two
closest selected items as far apart as possible. Classically the **p-dispersion** problem, and the
oldest and best-studied objective in the family. Its characteristic difficulty is that the score
depends on one pair only, so most [swaps](#swap) leave it unchanged and the solver needs
tie-breakers to make progress.

## Max-sum objective { #max-sum-objective }

Maximize the mean (equivalently, total) distance over all selected *pairs* — the objective the
literature calls **the** Maximum Diversity Problem, and the one with the largest body of published
benchmark results. It maximizes total spread and, unlike every separation-based objective, does
not penalize near-duplicates as such: two nearly identical items can both be kept if both sit far
from everything else.

## Overlapping constraints { #overlapping-constraints }

[Constraint groups](#constraint-group) that share items, so one item can count toward several
requirements at once — for example a demographic group and a geographic region. max-div supports
these directly, which is unusual: most alternative tools support either no selection constraints
or only a partition into disjoint strata. See [Constraints](constraints.md).

## Penalty shape { #penalty-shape }

How a constraint violation's size translates into lost constraint score — linear by default, or
quadratic. Linear treats every unit of violation alike; quadratic punishes one large violation far
more than several small ones, which pushes the solver toward spreading its shortfalls when it
cannot satisfy everything.

## Preset { #preset }

A named bundle of solver configuration — an initialization strategy, one or more optimization
strategies, and how long each runs — so that a working solver is one call rather than a tuning
exercise. The presets trade setup cost against answer quality and are the recommended starting
point; the underlying [solver steps](#solver-step) remain configurable for anyone who needs them.

## Score components { #score-components }

Every selection is scored on four components compared in **strict priority order**: selection
size, then [constraint](#feasibility) satisfaction, then diversity, then diversity tie-breakers. A
selection that is better on an earlier component wins regardless of how much worse it is on a
later one — the comparison is lexicographic, not a weighted sum. This is what lets the solver
pursue feasibility and diversity at once without either drowning out the other. See
[Scoring](scoring.md).

## Selection { #selection }

The subset of *k* [items](#item) currently chosen — what the solver maintains, modifies via
[swaps](#swap), and ultimately returns. *k* is fixed by the problem, so the solver's job is
choosing *which* items rather than how many.

## Separation { #separation }

For a selected item, the distance to its nearest neighbor *within the selection* — not within the
whole dataset. High separation means the item sits well away from everything else selected; low
separation means something else selected is close to it. Separation is the
[diversity contribution](#diversity-contribution) that the `*_SEPARATION` metrics aggregate, and it
is maintained incrementally as items are added and removed rather than recomputed after every
[swap](#swap). See [Diversity & Distance](diversity.md).

## Solver step { #solver-step }

One stage of the solving pipeline, with its own strategy and its own duration. A run is exactly one
**initialization step**, which builds the first [selection](#selection), followed by one or more
**optimization steps**, which improve it by [local search](#local-search). Splitting a run into
steps is what allows a fast, broad phase to be followed by a slower, more careful one — which is
how the [presets](#preset) are built. See [How the Solver Works](solver.md).

## Swap { #swap }

The single move of max-div's [local search](#local-search): remove one or more items from the
[selection](#selection), add the same number of replacements, and keep the result only if the
[score](#score-components) improves. How many items move at once is the *swap size*; some
strategies adapt it during the run, since small swaps stop paying off once the selection is
already good, while large ones cost more per iteration.

## Vector { #vector }

The coordinates of an [item](#item), when the problem has any. A vector-defined problem takes an
*n*×*d* array and computes distances from it with the chosen
[distance metric](#distance-metric); a problem built from a
[condensed distance matrix](#condensed-distance-matrix) has distances but no vectors. Use *vector*
only where the coordinates themselves matter — how distances are computed, or what gets passed in
— and [item](#item) everywhere the selection is what is being discussed.
