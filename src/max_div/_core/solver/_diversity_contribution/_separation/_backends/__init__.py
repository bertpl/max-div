"""One module per storage layout, and the lookup that picks the right one.

Each module provides the same three calculations over the signatures in `_signatures`, so the
modules are interchangeable by construction and a fourth layout is a module plus an entry below.

Which one to use is decided here, in Python, and never inside a compiled function. A layout test
inside one of these loops cannot be lifted out of it — one layout computes distances in a loop of
its own, which the compiler will not duplicate the outer loop to avoid — and the loop then runs one
item at a time, costing the stored layouts most of their speed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from max_div._core.metrics._distance._store import KIND_CONDENSED, KIND_FULL_MATRIX, KIND_LAZY

from . import _condensed, _full_matrix, _lazy

if TYPE_CHECKING:
    from collections.abc import Callable

    from max_div._core.metrics._distance import DistanceStore


class SeparationBackend(NamedTuple):
    """What the separation tracker needs from one storage layout.

    `elements` computes separations for the given items from scratch; `add` and `remove` update
    them after an item joins or leaves the selection.
    """

    elements: Callable[..., None]
    add: Callable[..., None]
    remove: Callable[..., None]


BACKEND_BY_KIND: dict[int, SeparationBackend] = {
    int(KIND_FULL_MATRIX): SeparationBackend(_full_matrix.elements, _full_matrix.add, _full_matrix.remove),
    int(KIND_CONDENSED): SeparationBackend(_condensed.elements, _condensed.add, _condensed.remove),
    int(KIND_LAZY): SeparationBackend(_lazy.elements, _lazy.add, _lazy.remove),
}


def backend_for(store: DistanceStore) -> SeparationBackend:
    """Return the separation calculations valid for this store's layout."""
    return BACKEND_BY_KIND[int(store.kind)]
