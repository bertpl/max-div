# Distance Storage

During search the solver reads pairwise distances constantly, and how they are stored is
selectable on the builder:

```python
from max_div.solver import DistanceStorageType

solver = (
    MaxDivSolverBuilder(problem)
    .with_preset(seconds(5))
    .with_distance_storage(DistanceStorageType.FULL_MATRIX)  # optional; AUTO is the default
    .build()
)
```

- **`FULL_MATRIX`** — a full `n x n` matrix of float32 values, so reading a distance is a contiguous
  row scan. A problem built via `from_distances` from a condensed vector is expanded into this
  layout, at twice the memory of the condensed input.
- **`LAZY`** — no stored distances at all: each distance is computed on demand from the vectors.
  Slower per distance, but removes the O(n²) memory requirement entirely, so much larger problems
  become feasible. Available only when the problem is built from vectors.
- **`AUTO`** (default) — for vector problems, the full matrix when it fits comfortably in memory
  and lazy otherwise; for problems built via `from_distances`, always the full matrix. The
  resolved storage type per distance is reported in the solution summary, e.g. `storage=full_matrix (L2)`
  — pin a storage type explicitly to override.

## Reproducibility

**On one machine, with the same installed versions — max-div's and numba's — and the same
backend, a seeded solve is exactly reproducible**: run it again and you get the same selection,
bit for bit.

**Change any of those three and you may get a different — equally diverse — selection.**
Distances are accumulated sums, and the compiler is allowed to reorder such a sum to vectorize it;
how it does so depends on the processor and on the numba version that compiled the distance functions, which
is why a numba upgrade counts here as much as a max-div one. The resulting differences are in the
last bits, but the search is a chaotic process, so one differing comparison can send it down a
different path to an equally good answer. The difference is not a slightly different selection —
it is a different one of comparable quality.

Two practical consequences:

- **`AUTO` picks a backend from available memory**, so the same problem can resolve differently on
  a machine with more or less RAM. Pin the backend explicitly if you want that variable removed —
  though on its own that does not make results portable across different machines.
- **Comparing runs meaningfully** means comparing achieved diversity, not selected indices.

(With a time budget, not an iteration budget, a faster backend also completes more
iterations — the machine-dependence any time budget carries.)
