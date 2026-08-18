import pytest

from max_div._core._cli.bm_solver_sizing import K_VALUES, determine_problem_size_for_k
from max_div._core.benchmark_problems import BenchmarkProblemFactory

ALL_PROBLEMS = BenchmarkProblemFactory.get_all_benchmark_names()


@pytest.mark.parametrize("problem_name", ALL_PROBLEMS)
@pytest.mark.parametrize("k_target", K_VALUES)
def test_determine_problem_size_for_k(problem_name: str, k_target: int):
    """Every problem resolves each k in K_VALUES to the largest n selecting exactly that many items."""
    # --- act --------------------------
    n = determine_problem_size_for_k(problem_name, k_target)

    # --- assert -----------------------
    _d, _n, k, _m, _n_con = BenchmarkProblemFactory.get_problem_dimensions(problem_name, n=n)
    _d, _n, k_next, _m, _n_con = BenchmarkProblemFactory.get_problem_dimensions(problem_name, n=n + 1)
    assert k == k_target
    assert k_next > k_target


def test_determine_problem_size_for_k_unreachable(monkeypatch: pytest.MonkeyPatch):
    """A k(n) mapping that skips the target raises instead of silently returning a nearby size."""

    # --- arrange ----------------------
    def _even_k_only(problem_name: str, n: int) -> tuple[int, int, int, int, int]:
        """Fake dimensions whose k(n) only takes even values."""
        return 2, n, 2 * ((n + 9) // 10), 0, 0

    monkeypatch.setattr(BenchmarkProblemFactory, "get_problem_dimensions", staticmethod(_even_k_only))

    # --- act & assert -----------------
    with pytest.raises(ValueError, match="no size n"):
        determine_problem_size_for_k("U1", k_target=101)
