"""One module per storage layout, and the lookup that picks the right one.

Each module provides the same three calculations over the signatures in `.._signatures`, so the
modules are interchangeable by construction and a fourth layout is a module plus an entry below.

Which one to use is decided here, in Python, and never inside a compiled function — see the
separation package for why a layout test cannot live inside one of these loops.

Each calculation covers a range excluding its own index, which keeps partners in ascending order
so sums stay bit-equal across layouts, and lets every read go through a distinct-items reader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from max_div._core.metrics._distance._store import KIND_CONDENSED, KIND_FULL_MATRIX, KIND_LAZY

from . import _condensed, _full_matrix, _lazy

if TYPE_CHECKING:
    from collections.abc import Callable

    from max_div._core.metrics._distance import DistanceStore


class MeanDistanceBackend(NamedTuple):
    """What the mean-distance tracker needs from one storage layout.

    `elements` computes mean distances for the given items from scratch; `add` and `remove` update
    the running sums after an item joins or leaves the selection.
    """

    elements: Callable[..., None]
    add: Callable[..., None]
    remove: Callable[..., None]


BACKEND_BY_KIND: dict[int, MeanDistanceBackend] = {
    int(KIND_FULL_MATRIX): MeanDistanceBackend(_full_matrix.elements, _full_matrix.add, _full_matrix.remove),
    int(KIND_CONDENSED): MeanDistanceBackend(_condensed.elements, _condensed.add, _condensed.remove),
    int(KIND_LAZY): MeanDistanceBackend(_lazy.elements, _lazy.add, _lazy.remove),
}


def backend_for(store: DistanceStore) -> MeanDistanceBackend:
    """Return the mean-distance calculations valid for this store's layout."""
    return BACKEND_BY_KIND[int(store.kind)]
