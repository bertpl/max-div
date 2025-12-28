from max_div.solver._strategies._initialization._time_model import InitializationTimeModel


def test_initialization_time_model_get_time_sec():
    # --- arrange -----------------------------------------
    t_model = InitializationTimeModel(
        t0=0.1,
        t0_k=0.01,
        t_n=0.001,
        t_k=0.0001,
        t_m=0.00001,
        t_n_con_indices=0.000001,
    )

    # --- act ---------------------------------------------
    t_est = t_model.get_time_sec(n=1, k=2, m=3, n_con_indices=4)

    # --- assert ------------------------------------------
    assert t_est == 0.1 + 2 * (0.01 + (0.001 * 1) + (0.0001 * 2) + (0.00001 * 3) + (0.000001 * 4))


def test_initialization_time_model_repr_str():
    # --- arrange -----------------------------------------
    t_model = InitializationTimeModel(
        t0=0.1,
        t0_k=0.01,
        t_n=0.001,
        t_k=0.0001,
        t_m=0.00001,
        t_n_con_indices=0.000001,
    )

    expected_result = (
        "InitializationTimeModel(t0=1.00e-01, t0_k=1.00e-02, t_n=1.00e-03, "
        + "t_k=1.00e-04, t_m=1.00e-05, t_n_con_indices=1.00e-06)"
    )

    # --- act ---------------------------------------------
    str_result = str(t_model)
    repr_result = repr(t_model)

    # --- assert ------------------------------------------
    assert expected_result == str_result
    assert expected_result == repr_result
