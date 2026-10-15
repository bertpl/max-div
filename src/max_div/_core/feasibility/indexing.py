"""`build_item_constraint_csr` transposes the packed constraint->item layout into an item->constraint CSR index."""

import numpy as np
from numpy.typing import NDArray

from max_div._core.jit import lazy_njit


@lazy_njit("Tuple((int64[::1], int32[::1]))(int32[::1], int64)", cache=True)
def build_item_constraint_csr(con_indices: NDArray[np.int32], n: int) -> tuple[NDArray[np.int64], NDArray[np.int32]]:
    """Transpose the packed constraint->item layout into an item->constraint CSR (compressed sparse row) index.

    Args:
        con_indices: the packed representation built by `to_numpy_constraints` — a 2m-element
            header of per-constraint [start, end) offsets, followed by the concatenated member
            indices.
        n: the number of items.

    Returns:
        `(item_indptr, item_cons)`: for item `j`, `item_cons[item_indptr[j]:item_indptr[j + 1]]`
        lists the constraints containing `j`.
    """
    item_indptr = np.zeros(n + 1, dtype=np.int64)
    if con_indices.shape[0] == 0:
        return item_indptr, np.empty(0, dtype=np.int32)
    m = con_indices[0] // 2

    # count memberships per item, then prefix-sum into indptr
    for e in range(2 * m, con_indices.shape[0]):
        item_indptr[con_indices[e] + 1] += 1
    for j in range(n):
        item_indptr[j + 1] += item_indptr[j]

    # fill, walking the constraints in order
    item_cons = np.empty(con_indices.shape[0] - 2 * m, dtype=np.int32)
    cursor = item_indptr[:-1].copy()
    for i in range(m):
        for e in range(con_indices[2 * i], con_indices[2 * i + 1]):
            j = con_indices[e]
            item_cons[cursor[j]] = np.int32(i)
            cursor[j] += 1
    return item_indptr, item_cons
