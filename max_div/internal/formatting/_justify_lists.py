def ljust_str_list(str_list: list[str]) -> list[str]:
    """Left-justify all strings in the given list to the length of the longest string."""
    if len(str_list) == 0:
        return []
    else:
        max_len = max(len(s) for s in str_list)
        return [s.ljust(max_len) for s in str_list]


def rjust_str_list(str_list: list[str]) -> list[str]:
    """Right-justify all strings in the given list to the length of the longest string."""
    if len(str_list) == 0:
        return []
    else:
        max_len = max(len(s) for s in str_list)
        return [s.rjust(max_len) for s in str_list]
