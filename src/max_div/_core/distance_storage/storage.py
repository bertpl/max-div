"""Define the user's storage-type choice and the per-store types that a solve resolves to."""

from dataclasses import dataclass
from enum import StrEnum


class DistanceStorageType(StrEnum):
    """A `DistanceStorageType` names how the solver stores pairwise distances during search.

    `AUTO` (the default) lets max-div decide; `DistanceStoragePlan._decide_storage_types` states the
    policy.  The resolved storage type is reported in the solution summary.

    Pinning a storage type overrides the policy — `LAZY` requires vectors, so it is unavailable for
    distance-input problems.  A condensed distance input is expanded to the full matrix, at twice its
    memory.
    """

    AUTO = "auto"
    FULL_MATRIX = "full_matrix"
    LAZY = "lazy"


@dataclass(frozen=True)
class DistanceStorageTypes:
    """Record how each of a solve's distance stores was stored: its label paired with the resolved storage type.

    One entry per store, in store order.  The label is the store's distance spec label: a metric's
    label such as `L2`, or `user distances` for the distances that a distance-input problem was given.
    """

    per_store: tuple[tuple[str, DistanceStorageType], ...] = ()

    def __str__(self) -> str:
        """Group by storage type, listing each type's labels: `full_matrix (L1, L2), lazy (geomean)`."""
        labels_by_type: dict[DistanceStorageType, list[str]] = {}
        for label, storage_type in self.per_store:
            labels_by_type.setdefault(storage_type, []).append(label)
        return ", ".join(
            f"{storage_type.value} ({', '.join(labels)})" for storage_type, labels in labels_by_type.items()
        )
