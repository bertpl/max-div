from max_div.solver._solver_state import SolverState

from ._base import OptimizationStrategy


class OptimDummy(OptimizationStrategy):
    def _perform_single_iteration(self, state: SolverState, progress_frac: float):
        pass  # TODO
