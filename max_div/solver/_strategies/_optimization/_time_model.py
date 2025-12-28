class OptimizationTimeModel:
    """
    Class to estimate duration of a single iteration of an optimization strategy based on problem parameters.

        t_per_iter = t0 + (t_n * n) + (t_k * k) + (t_m * m) + (t_n_con_indices * n_con_indices)
    """

    def __init__(self, t0: float, t_n: float, t_k: float, t_m: float, t_n_con_indices: float):
        self.t0 = t0
        self.t_n = t_n
        self.t_k = t_k
        self.t_m = t_m
        self.t_n_con_indices = t_n_con_indices

    def get_time_sec(self, n: int, k: int, m: int, n_con_indices: int) -> float:
        """Calculate estimated duration (sec) of a single optimization iteration based on problem parameters."""
        return self.t0 + (self.t_n * n) + (self.t_k * k) + (self.t_m * m) + (self.t_n_con_indices * n_con_indices)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f"OptimizationTimeModel(t0={self.t0:.2e}, t_n={self.t_n:.2e}, "
            + f"t_k={self.t_k:.2e}, t_m={self.t_m:.2e}, t_n_con_indices={self.t_n_con_indices:.2e})"
        )
