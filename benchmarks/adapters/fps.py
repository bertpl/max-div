"""Farthest-point-sampling adapters: fpsample (Rust) and skmatter (sklearn-contrib)."""

import numpy as np
from numpy.typing import NDArray

from max_div.problem import MaxDivProblem

from ._inputs import problem_vectors
from .base import SelectionAdapter


class FpsampleFPS(SelectionAdapter):
    """Greedy farthest-point sampling via the fpsample package (max-min, 2-approx).

    `variant` picks the algorithm: `"vanilla"` is the plain greedy sweep; `"kdline"` is the
    bucket KD-line (QuickFPS) variant, tree-accelerated and well suited to low dimensions.
    """

    def __init__(self, variant: str = "vanilla") -> None:
        self.variant = variant

    @property
    def name(self) -> str:
        """Tool name as it appears in records and figures."""
        return "fpsample[FPS]" if self.variant == "vanilla" else "fpsample[KDLine]"

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Run the chosen FPS variant, seeding via the start index."""
        import fpsample

        vectors = problem_vectors(problem)
        start_idx = seed % problem.n
        if self.variant == "kdline":
            # 2**h buckets must not exceed n; cap at 7, the depth the low-d recommendation uses
            height = min(7, max(1, int(np.floor(np.log2(problem.n)))))
            selected = fpsample.bucket_fps_kdline_sampling(vectors, problem.k, height, start_idx=start_idx)
        else:
            selected = fpsample.fps_sampling(vectors, problem.k, start_idx=start_idx)
        return np.asarray(selected, dtype=np.int64)


class SkmatterFPS(SelectionAdapter):
    """Greedy farthest-point sampling via skmatter's FPS selector (max-min, 2-approx)."""

    @property
    def name(self) -> str:
        """Tool name as it appears in records and figures."""
        return "skmatter[FPS]"

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Run skmatter FPS, seeding via the initial point."""
        from skmatter.sample_selection import FPS

        vectors = problem_vectors(problem)
        selector = FPS(n_to_select=problem.k, initialize=seed % problem.n)
        selector.fit(vectors)
        return np.asarray(selector.selected_idx_, dtype=np.int64)
