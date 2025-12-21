class ProgressTable:
    def __init__(self, headers: list[str]):
        self.headers = headers

    def show_header(self):
        self.print_row(self.headers)
        self.print_row(["-" * len(h) for h in self.headers])

    def show_progress(self, values: list[str]):
        values_adjusted = []
        for value, header in zip(values, self.headers):
            target_len = len(header)
            value = value.rjust(target_len)[:target_len]
            values_adjusted.append(value)

        self.print_row(values_adjusted)

    @staticmethod
    def print_row(values: list[str], sep="|"):
        print(sep + sep.join([f" {v} " for v in values]) + sep)
