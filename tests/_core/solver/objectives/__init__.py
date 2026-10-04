"""Build the objectives that solver tests hand to `SolverState.new` and `ScoreGenerator`, and their distances."""

import numpy as np
from numpy.typing import NDArray

from max_div._core.distance_storage import AdoptingDataMatrixProducer, InProcessDataMatrixReader
from max_div._core.metrics import DistanceMetric, DiversityMetric, DiversityObjective, DiversityObjectiveSimple
from max_div._core.metrics._distance import FullMatrixDistanceSpec, compute_full_matrix

# Every objective that these helpers build reads one full distance matrix, which this spec names.
TEST_DISTANCE_SPEC = FullMatrixDistanceSpec(matrix_id=0, label="test distances")


def simple_objective(diversity_metric: DiversityMetric) -> DiversityObjectiveSimple:
    """Return a simple objective of `diversity_metric` over `TEST_DISTANCE_SPEC`."""
    return DiversityObjectiveSimple(diversity_metric, TEST_DISTANCE_SPEC)


def tie_breaker_objectives(tie_breaker_metrics: list[DiversityMetric]) -> list[DiversityObjective]:
    """Return each metric as a simple tie-breaker objective over `TEST_DISTANCE_SPEC`.

    A single-metric problem's tie-breakers are simple objectives, as the solver builder constructs them.
    """
    return [DiversityObjectiveSimple(metric, TEST_DISTANCE_SPEC) for metric in tie_breaker_metrics]


def full_matrix_reader(vectors: NDArray[np.float32], metric: DistanceMetric) -> InProcessDataMatrixReader:
    """Return a reader whose data matrix under `TEST_DISTANCE_SPEC` is the full distance matrix of `vectors`."""
    return InProcessDataMatrixReader(
        {TEST_DISTANCE_SPEC.matrix_id: AdoptingDataMatrixProducer(compute_full_matrix(vectors, metric))}
    )
