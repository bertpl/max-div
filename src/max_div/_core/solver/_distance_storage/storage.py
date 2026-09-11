"""The user's storage-type choice and the per-store types a solve resolved to."""

from dataclasses import dataclass
from enum import StrEnum

from max_div._core.metrics._distance import DistanceMetric


class DistanceStorageType(StrEnum):
    """A `DistanceStorageType` names how the solver stores pairwise distances during search.

    `AUTO` (the default) lets max-div decide; `DistanceStoreFactory.determine_storage_types` states
    the policy.  The resolved storage type is reported in the solution summary.  Pinning a storage type
    overrides the policy — `LAZY` requires vectors, so it is unavailable for distance-input
    problems.  A condensed distance input is expanded to the full matrix, at twice its memory.
    """

    AUTO = "auto"
    FULL_MATRIX = "full_matrix"
    LAZY = "lazy"


@dataclass(frozen=True)
class DistanceStorageTypes:
    """How each of a solve's distance stores was stored: its distance paired with the resolved storage type.

    One entry per store, in store order.  A distance-input problem's store carries `None` for its
    distance, since it holds given distances rather than a metric.
    """

    per_store: tuple[tuple[DistanceMetric | None, DistanceStorageType], ...] = ()

    def __str__(self) -> str:
        """Group by storage type, listing each type's distance labels: `full_matrix (L1, L2), lazy (geomean)`."""
        metrics_by_type: dict[DistanceStorageType, list[str]] = {}
        for distance, storage_type in self.per_store:
            labels = metrics_by_type.setdefault(storage_type, [])
            if distance is not None:
                labels.append(distance.label)
        return ", ".join(
            f"{storage_type.value} ({', '.join(labels)})" if labels else storage_type.value
            for storage_type, labels in metrics_by_type.items()
        )
