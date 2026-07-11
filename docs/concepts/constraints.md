# Constraints

## What Are Constraints?

Constraints enforce that the selected subset includes a minimum and/or maximum number of vectors
from specific groups. Each constraint is defined by:

- **`int_set`** -- a set of vector indices that form the group
- **`min_count`** -- minimum number of vectors to select from this group
- **`max_count`** -- maximum number of vectors to select from this group
- **`weight`** -- how strongly this constraint counts toward feasibility (default `1`, must be `> 0`)

```python
from max_div import Constraint

# At least 3 and at most 7 vectors from indices 0-49
Constraint(int_set=set(range(50)), min_count=3, max_count=7)
```

## Weighting Constraints

By default every constraint counts equally toward the feasibility score. Raise a constraint's
`weight` above `1` to make the solver prioritize satisfying it over lower-weight constraints when
they cannot all be satisfied at once:

```python
constraints = [
    Constraint(int_set=must_have_indices, min_count=2, max_count=5, weight=3.0),  # prioritized
    Constraint(int_set=nice_to_have_indices, min_count=1, max_count=4),           # weight 1
]
```

A violation of a weight-3 constraint counts three times as much as an equal violation of a weight-1
constraint. Weights must be strictly positive -- a zero weight would let a genuine violation go
unpenalized and be reported as feasible.

## Penalizing Large Violations

By default, a constraint's contribution to the score grows *linearly* with the size of its
violation. Switch to *quadratic* penalization to push the solver harder away from large individual
violations (preferring several small violations over one large one):

```python
from max_div import MaxDivSolverBuilder
from max_div.solver import ConstraintPenalty

solver = (
    MaxDivSolverBuilder(problem)
    .with_constraint_penalty(ConstraintPenalty.QUADRATIC)  # default is ConstraintPenalty.LINEAR
    .with_preset(seconds(5))
    .build()
)
```

This is a single global setting for the whole solve, independent of any per-constraint weights.

## Overlapping Constraints

A single vector can belong to multiple constraint groups. This is useful for modeling
multi-dimensional fairness requirements.

For example, you might have demographic groups and geographic groups:

```python
constraints = [
    # Demographic constraints
    Constraint(int_set=group_a_indices, min_count=5, max_count=10),
    Constraint(int_set=group_b_indices, min_count=5, max_count=10),
    # Geographic constraints
    Constraint(int_set=region_north,    min_count=3, max_count=8),
    Constraint(int_set=region_south,    min_count=3, max_count=8),
]
```

A vector can be in both `group_a_indices` and `region_north`, and both constraints will
be tracked independently.

## Infeasible Constraints

Constraints can be infeasible -- either individually or jointly. For example:

- **Individually infeasible**: `min_count=15` on a group with only 10 members
- **Jointly infeasible**: two non-overlapping groups each requiring `min_count=11` when `k=20`

The solver does **not** fail on infeasible constraints. Instead, it finds the *least infeasible*
solution -- the one with the smallest total constraint violation. This is a natural consequence
of the score's priority ordering: the solver always prefers better constraint satisfaction over
higher diversity.

## How Constraints Are Tracked

Internally, constraints are represented as numpy arrays for efficient access within
numba-compiled solver code. For each constraint, the solver tracks how many *additional*
selections are needed (`min_remaining`) and how many more are *allowed* (`max_remaining`):

- A constraint is **satisfied** when `min_remaining <= 0` and `max_remaining >= 0`
- Adding a vector from a constraint group decreases both counters by 1
- Removing a vector increases both counters by 1

This incremental tracking avoids recomputing constraint satisfaction from scratch after each
swap, keeping the per-iteration cost constant regardless of constraint complexity.

## Impact on Diversity

Adding constraints restricts the search space, which generally reduces the achievable diversity.
The tighter the constraints, the more diversity is sacrificed. This is expected and visible in
the solution's score: a constrained solution will typically have a lower `diversity` value than
an equivalent unconstrained one.
