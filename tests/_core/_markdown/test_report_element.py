import pytest

from max_div._core._markdown import ReportHeader, ReportText, h1, h2, h3, h4, text


# =================================================================================================
#  ReportText
# =================================================================================================
@pytest.mark.parametrize(
    "txt, markdown, expected_lines",
    [
        ("test", False, ["test"]),
        ("test", True, ["test"]),
        ("line1\nline2", False, ["line1", "line2"]),
        ("line1\nline2", True, ["line1<br>line2"]),
        ("This is `code` example.", False, ["This is 'code' example."]),
        ("This is `code` example.", True, ["This is `code` example."]),
    ],
)
def test_report_text(txt: str, markdown: bool, expected_lines: list[str]):
    # --- arrange ----------------------
    report_text = ReportText(txt)

    # --- act --------------------------
    lines = report_text.render(markdown=markdown)

    # --- assert -----------------------
    assert isinstance(lines, list)
    assert len(lines) == len(expected_lines)
    for line, expected_line in zip(lines, expected_lines):
        assert line == expected_line


def test_report_text_alias():
    # --- act --------------------------
    report_text = text("test")

    # --- assert -----------------------
    assert isinstance(report_text, ReportText)
    assert report_text.txt == "test"


# =================================================================================================
#  ReportHeader
# =================================================================================================
@pytest.mark.parametrize(
    "txt, level, markdown, expected_lines",
    [
        ("Header 1", 1, True, ["", "# Header 1", ""]),
        ("Header 1", 1, False, ["", "HEADER 1", ""]),
        ("Header 2", 2, True, ["", "## Header 2", ""]),
        ("Header 2", 2, False, ["", "Header 2", ""]),
        ("Header 3", 3, True, ["", "### Header 3", ""]),
        ("Header 3", 3, False, ["", "Header 3", ""]),
        ("Header 4", 4, True, ["", "#### Header 4", ""]),
        ("Header 4", 4, False, ["", "Header 4", ""]),
        ("This is `code` header.", 1, True, ["", "# This is `code` header.", ""]),
        ("This is `code` header.", 1, False, ["", "THIS IS 'CODE' HEADER.", ""]),
        ("This is `code` header.", 2, False, ["", "This is 'code' header.", ""]),
    ],
)
def test_report_header(txt: str, level: int, markdown: bool, expected_lines: list[str]):
    # --- arrange ----------------------
    report_header = ReportHeader(txt, level=level)

    # --- act --------------------------
    lines = report_header.render(markdown=markdown)

    # --- assert -----------------------
    assert isinstance(lines, list)
    assert len(lines) == len(expected_lines)
    for line, expected_line in zip(lines, expected_lines):
        assert line == expected_line


def test_report_header_aliases():
    # --- act --------------------------
    h1_element = h1("Header 1")
    h2_element = h2("Header 2")
    h3_element = h3("Header 3")
    h4_element = h4("Header 4")

    # --- assert -----------------------
    assert isinstance(h1_element, ReportHeader)
    assert h1_element.txt == "Header 1"
    assert h1_element.level == 1

    assert isinstance(h2_element, ReportHeader)
    assert h2_element.txt == "Header 2"
    assert h2_element.level == 2

    assert isinstance(h3_element, ReportHeader)
    assert h3_element.txt == "Header 3"
    assert h3_element.level == 3

    assert isinstance(h4_element, ReportHeader)
    assert h4_element.txt == "Header 4"
    assert h4_element.level == 4
