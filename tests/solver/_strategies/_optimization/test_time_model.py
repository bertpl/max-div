from max_div.solver._strategies._optimization._time_model import OptimizationTimeModel


def test_optimization_time_model_get_time_sec():
    # --- arrange -----------------------------------------
    t_model = OptimizationTimeModel(
        t0=0.2,
        t_n=0.02,
        t_k=0.002,
        t_m=0.0002,
        t_n_con_indices=0.00002,
    )

    # --- act ---------------------------------------------
    t_est = t_model.get_time_sec(n=1, k=2, m=3, n_con_indices=4)

    # --- assert ------------------------------------------
    assert t_est == 0.2 + (0.02 * 1) + (0.002 * 2) + (0.0002 * 3) + (0.00002 * 4)


def test_optimization_time_model_repr_str():
    # --- arrange -----------------------------------------
    t_model = OptimizationTimeModel(
        t0=0.2,
        t_n=0.02,
        t_k=0.002,
        t_m=0.0002,
        t_n_con_indices=0.00002,
    )

    expected_result = (
        "OptimizationTimeModel(t0=2.00e-01, t_n=2.00e-02, t_k=2.00e-03, " + "t_m=2.00e-04, t_n_con_indices=2.00e-05)"
    )

    # --- act ---------------------------------------------
    str_result = str(t_model)
    repr_result = repr(t_model)

    # --- assert ------------------------------------------
    assert expected_result == str_result
    assert expected_result == repr_result
