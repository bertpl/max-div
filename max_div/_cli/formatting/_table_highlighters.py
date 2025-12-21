from abc import ABC, abstractmethod

from max_div.internal.formatting import md_bold, md_colored

from ._content_types import BenchmarkResult, CellContent, NumberWithUncertainty, Percentage


# =================================================================================================
#  Markdown highlighters
# =================================================================================================
class HighLighter(ABC):
    @abstractmethod
    def process_row(self, row: list[CellContent]) -> list[CellContent]:
        raise NotImplementedError()


class FastestBenchmark(HighLighter):
    def __init__(self, bold: bool = True, color: str = "#00aa00"):
        self.bold = bold
        self.color = color

    def process_row(self, row: list[CellContent]) -> list[CellContent]:
        if any(isinstance(value, BenchmarkResult) for value in row):
            # Find the fastest BenchmarkResult (minimum median time)
            t_q50_min = min([value.t_sec_q_50 for value in row if isinstance(value, BenchmarkResult)])

            # Convert row to strings, highlighting the results with t_q25 <= t_q50_min
            converted_row: list[CellContent] = []
            for i, value in enumerate(row):
                if isinstance(value, BenchmarkResult):
                    text = str(value)
                    if value.t_sec_q_25 <= t_q50_min:
                        if self.bold:
                            text = md_bold(text)
                        text = md_colored(text, self.color)
                    converted_row.append(text)
                else:
                    converted_row.append(value)
            return converted_row
        else:
            return row


class HighestPercentage(HighLighter):
    def __init__(self, bold: bool = True, color: str = "#00aa00"):
        self.bold = bold
        self.color = color

    def process_row(self, row: list[CellContent]) -> list[CellContent]:
        if any(isinstance(value, Percentage) for value in row):
            # Find the highest Percentage (maximum frac)
            max_perc = max([value for value in row if isinstance(value, Percentage)], key=lambda x: x.frac)

            # Convert row to strings, highlighting the results with frac == max_frac
            converted_row: list[CellContent] = []
            for i, value in enumerate(row):
                if isinstance(value, Percentage):
                    text = str(value)
                    if text == str(max_perc):  # make green if its str-representation is equal
                        if self.bold:
                            text = md_bold(text)
                        text = md_colored(text, self.color)
                    converted_row.append(text)
                else:
                    converted_row.append(value)
            return converted_row
        else:
            return row


class LowestPercentage(HighLighter):
    def __init__(self, bold: bool = True, color: str = "#00aa00"):
        self.bold = bold
        self.color = color

    def process_row(self, row: list[CellContent]) -> list[CellContent]:
        if any(isinstance(value, Percentage) for value in row):
            # Find the highest Percentage (maximum frac)
            min_perc = min([value for value in row if isinstance(value, Percentage)], key=lambda x: x.frac)

            # Convert row to strings, highlighting the results with frac == min_frac
            converted_row: list[CellContent] = []
            for i, value in enumerate(row):
                if isinstance(value, Percentage):
                    text = str(value)
                    if text == str(min_perc):  # make green if its str-representation is equal
                        if self.bold:
                            text = md_bold(text)
                        text = md_colored(text, self.color)
                    converted_row.append(text)
                else:
                    converted_row.append(value)
            return converted_row
        else:
            return row


class HighestNumberWithUncertainty(HighLighter):
    def __init__(self, bold: bool = True, color: str = "#00aa00"):
        self.bold = bold
        self.color = color

    def process_row(self, row: list[CellContent]) -> list[CellContent]:
        if any(isinstance(value, NumberWithUncertainty) for value in row):
            # Find the highest NumberWithUncertainty (maximum value_q_50)
            q50_max = max([value.value_q_50 for value in row if isinstance(value, NumberWithUncertainty)])

            # Convert row to strings, highlighting the results with value_q_50 == max value_q_50
            converted_row: list[CellContent] = []
            for i, value in enumerate(row):
                if isinstance(value, NumberWithUncertainty):
                    text = str(value)
                    if value.value_q_75 >= q50_max:
                        if self.bold:
                            text = md_bold(text)
                        text = md_colored(text, self.color)
                    converted_row.append(text)
                else:
                    converted_row.append(value)
            return converted_row
        else:
            return row


class BoldLabels(HighLighter):
    def process_row(self, row: list[CellContent]) -> list[CellContent]:
        converted_row: list[CellContent] = []
        for value in row:
            if isinstance(value, str) and value.endswith(":"):
                converted_row.append(md_bold(value))
            else:
                converted_row.append(value)
        return converted_row
