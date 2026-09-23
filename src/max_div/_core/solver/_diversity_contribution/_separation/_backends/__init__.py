"""One module per storage layout, and the lookup that picks the right one.

Each module provides the same calculations over the signatures in `._signatures`, so the
modules are interchangeable by construction and another layout is a module plus an entry below.

Which one to use is decided here, in Python, and never inside a compiled function. A layout test
inside one of these loops cannot be lifted out of it — one layout computes distances in a loop of
its own, which the compiler will not duplicate the outer loop to avoid — and the loop then runs one
item at a time, costing the stored layouts most of their speed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from max_div._core.metrics._distance._store import KIND_FULL_MATRIX, KIND_LAZY

from . import _full_matrix, _lazy

if TYPE_CHECKING:
    from collections.abc import Callable

    from max_div._core.metrics._distance import DistanceStore


class SeparationBackend(NamedTuple):
    """What the separation tracker needs from one storage layout.

    - `add`, `add_many` and `remove` update the separations after items join or leave the selection.
    - `add_many` handles a batch in one pass over all items: it takes the minimum over the same
      distance pairs as sequential adds, so the result is identical, but the separation array `sep`
      is read and written once.
    - The `parallel` flag of `add_many` runs that pass over parallel threads; each item's separation
      is updated independently of the others, and a minimum does not depend on the order of its
      inputs, so results are identical there too.
    """

    add: Callable[..., None]
    add_many: Callable[..., None]
    remove: Callable[..., None]


BACKEND_BY_KIND: dict[int, SeparationBackend] = {
    int(KIND_FULL_MATRIX): SeparationBackend(_full_matrix.add, _full_matrix.add_many, _full_matrix.remove),
    int(KIND_LAZY): SeparationBackend(_lazy.add, _lazy.add_many, _lazy.remove),
}


def backend_for(store: DistanceStore) -> SeparationBackend:
    """Return the separation calculations valid for this store's layout."""
    return BACKEND_BY_KIND[int(store.kind)]
