import pytest

from max_div._core._markdown import TextLayout


@pytest.mark.parametrize(
    "txt, layout, expected_result",
    [
        ("test", TextLayout(), "test"),
        ("test", TextLayout(italic=True), "*test*"),
        ("test", TextLayout(bold=True), "**test**"),
        ("test", TextLayout(bold=True, italic=True), "***test***"),
        ("test", TextLayout(color="#ff0000"), '<span style="color:#ff0000">test</span>'),
        ("test", TextLayout(bold=True, color="#00ff00"), '<span style="color:#00ff00">**test**</span>'),
        ("line1<br>line2", TextLayout(bold=True), "**line1**<br>**line2**"),
        ("line1<br>line2<br>", TextLayout(bold=True), "**line1**<br>**line2**<br>"),
        ("", TextLayout(bold=True, italic=True, color="#0000ff"), ""),
    ],
)
def test_text_layout(txt: str, layout: TextLayout, expected_result: str):
    # --- act ---------------------------------------------
    txt_with_layout = layout.apply(txt)

    # --- assert ------------------------------------------
    assert txt_with_layout == expected_result
