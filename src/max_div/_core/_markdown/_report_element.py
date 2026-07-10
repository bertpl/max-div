from abc import ABC, abstractmethod


# =================================================================================================
#  Base class
# =================================================================================================
class ReportElement(ABC):
    @abstractmethod
    def render(self, markdown: bool) -> list[str]:
        raise NotImplementedError()


# =================================================================================================
#  Some common elements
# =================================================================================================
class ReportText(ReportElement):
    def __init__(self, txt: str):
        self.txt = txt

    def render(self, markdown: bool) -> list[str]:
        if markdown:
            lines = ["<br>".join(self.txt.splitlines())]
        else:
            txt = self.txt.replace("`", "'")
            lines = txt.splitlines()

        return lines


class ReportHeader(ReportElement):
    def __init__(self, txt: str, level: int = 1):
        self.txt = txt
        self.level = level

    def render(self, markdown: bool) -> list[str]:
        if markdown:
            lines = ["", f"{'#' * self.level} {self.txt}", ""]
        else:
            txt = self.txt.replace("`", "'")
            if self.level == 1:
                lines = ["", txt.upper(), ""]
            else:
                lines = ["", txt, ""]

        return lines


# =================================================================================================
#  Shortcuts
# =================================================================================================
def text(txt: str) -> ReportText:
    return ReportText(txt)


def h1(txt: str) -> ReportHeader:
    return ReportHeader(txt, level=1)


def h2(txt: str) -> ReportHeader:
    return ReportHeader(txt, level=2)


def h3(txt: str) -> ReportHeader:
    return ReportHeader(txt, level=3)


def h4(txt: str) -> ReportHeader:
    return ReportHeader(txt, level=4)
