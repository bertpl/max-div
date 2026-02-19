from contextlib import redirect_stdout
from io import StringIO

from max_div._core._markdown import Report, h1, h2, text


def test_report():
    # --- arrange -----------------------------------------
    report = Report()
    report.add(h1("Main Title about `code`"))
    report.add(h2("Subtitle"))
    report.add(text("Line 1."))
    report.add(["Line 2.", text("Line 3.")])
    report += "Line 4."
    report += ["Line 5.", "Line 6."]
    report += ["", "", ""]

    # --- act ---------------------------------------------
    md_report = report.render(markdown=True)
    txt_report = report.render(markdown=False)

    # Capture print() output with markdown=True
    md_print_buffer = StringIO()
    with redirect_stdout(md_print_buffer):
        report.print(markdown=True)
    md_print_output = md_print_buffer.getvalue().splitlines()

    # Capture print() output with markdown=False
    txt_print_buffer = StringIO()
    with redirect_stdout(txt_print_buffer):
        report.print(markdown=False)
    txt_print_output = txt_print_buffer.getvalue().splitlines()

    # --- assert ------------------------------------------
    assert md_report == [
        "# Main Title about `code`",
        "",
        "## Subtitle",
        "",
        "Line 1.",
        "Line 2.",
        "Line 3.",
        "Line 4.",
        "Line 5.",
        "Line 6.",
    ]

    assert txt_report == [
        "MAIN TITLE ABOUT 'CODE'",
        "",
        "Subtitle",
        "",
        "Line 1.",
        "Line 2.",
        "Line 3.",
        "Line 4.",
        "Line 5.",
        "Line 6.",
    ]

    # Check that print() and render() produce identical output
    assert md_print_output == md_report
    assert txt_print_output == txt_report
