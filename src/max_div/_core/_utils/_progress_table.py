class ProgressTable:
    def __init__(self, headers: list[str]) -> None:
        self.headers = headers

    def show_header(self) -> None:
        self.print_row(self.headers)
        self.print_line()

    def show_progress(self, values: list[str]) -> None:
        values_adjusted = []
        for value, header in zip(values, self.headers):
            target_len = len(header)
            values_adjusted.append(value.rjust(target_len)[:target_len])

        self.print_row(values_adjusted)

    def print_line(self) -> None:
        self.print_row(["-" * len(h) for h in self.headers])

    @staticmethod
    def print_row(values: list[str], sep: str = "|") -> None:
        print(sep + sep.join([f" {v} " for v in values]) + sep)
