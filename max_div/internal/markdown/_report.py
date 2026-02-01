from __future__ import annotations

from ._report_element import ReportElement, ReportText


class Report:
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self):
        self._elements: list[ReportElement] = []

    # -------------------------------------------------------------------------
    #  Build Report
    # -------------------------------------------------------------------------
    def add(self, elements: ReportElement | str | list[ReportElement | str]):
        if not isinstance(elements, list):
            elements = [elements]
        elements = [ReportText(el) if isinstance(el, str) else el for el in elements]
        self._elements.extend(elements)

    def __iadd__(self, elements: ReportElement | str | list[ReportElement | str]) -> Report:
        self.add(elements)
        return self

    # -------------------------------------------------------------------------
    #  Render Report
    # -------------------------------------------------------------------------
    def render(self, markdown: bool) -> list[str]:
        # --- render all elements -------------------------
        lines: list[str] = []
        for element in self._elements:
            lines.extend(element.render(markdown))

        # --- remove double empty lines -------------------
        cleaned_lines: list[str] = []
        previous_was_empty = False
        for line in lines:
            is_empty = line.strip() == ""
            if not (is_empty and previous_was_empty):
                cleaned_lines.append(line)
            previous_was_empty = is_empty

        # --- remove initial and final empty lines --------
        while cleaned_lines and cleaned_lines[0].strip() == "":
            cleaned_lines.pop(0)
        while cleaned_lines and cleaned_lines[-1].strip() == "":
            cleaned_lines.pop()

        # --- done ----------------------------------------
        return cleaned_lines

    def print(self, markdown: bool):
        for line in self.render(markdown):
            print(line)
