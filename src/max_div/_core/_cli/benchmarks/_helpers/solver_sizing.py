from max_div._core.benchmark_problems import MIN_N, BenchmarkProblemFactory

# The k values every solver-benchmark family (initialization, optimization, feasibility) sweeps,
# so a per-size row is comparable across problems even though they derive k from n differently
# (k = n/10 vs n/15): k, not n, drives the swap space and per-iteration cost the pages measure.
# The low end keeps the smallest problem small, so quick high-speed runs stay cheap; the top end
# reaches far enough that a constrained problem's feasible->infeasible crossing falls inside the
# table.
K_VALUES = [10, 20, 50, 100, 200, 500, 1000, 2000]


def determine_problem_size_for_k(problem_name: str, k_target: int) -> int:
    """Return the largest problem size n at which the problem selects exactly `k_target` items.

    Each benchmark problem derives k from n (see the docs problem-overview table); this inverts
    that derivation by bisecting on the monotone k(n).  Taking the largest such n lands on the
    round sizes (k=100 -> n=1000 for k=n/10 problems, n=1500 for k=n/15 ones).

    Raises:
        ValueError: If no n yields exactly `k_target` (a non-contiguous k(n) mapping).
    """

    def _k(n: int) -> int:
        """Return the problem's derived selection size k at size n."""
        _d, _n, k, _m, _n_con = BenchmarkProblemFactory.get_problem_dimensions(problem_name, n=n)
        return k

    # --- bracket --------------------------------
    # n_lo is the smallest constructible size; n_hi over-shoots k_target for any problem selecting
    # k >= n/100, which the built-in n/10 and n/15 problems clear with wide margin.
    n_lo, n_hi = MIN_N, 100 * k_target

    # --- bisect for largest n with k(n) <= k_target ---
    while n_hi - n_lo > 1:
        mid = (n_lo + n_hi) // 2
        if _k(mid) <= k_target:
            n_lo = mid
        else:
            n_hi = mid

    if _k(n_lo) != k_target:
        raise ValueError(f"Problem '{problem_name}' has no size n with k == {k_target}.")
    return n_lo
