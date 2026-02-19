from max_div._core._utils._justify_lists import ljust_str_list, rjust_str_list


def test_ljust_str_list():
    assert ljust_str_list([]) == []
    assert ljust_str_list(["a", "bb", "ccc"]) == ["a  ", "bb ", "ccc"]


def test_rjust_str_list():
    assert rjust_str_list([]) == []
    assert rjust_str_list(["a", "bb", "ccc"]) == ["  a", " bb", "ccc"]
