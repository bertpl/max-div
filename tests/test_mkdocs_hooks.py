"""Guards for the documentation-site build hooks.

The README's image paths are repo-root-relative so that GitHub resolves them; the docs build
re-anchors them onto the rendered page. The raster figures of every other page get a `width` at
their design size. These tests pin both rewrites, since nothing else fails loudly when they stop
happening — a broken or mis-sized image is invisible to the docs build.
"""

import importlib.util
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOKS = REPO_ROOT / "scripts" / "mkdocs_hooks.py"
README = REPO_ROOT / "README.md"


def _load_hooks():
    """Import the hook module by path — `scripts/` is maintainer tooling, not an importable package."""
    spec = importlib.util.spec_from_file_location("mkdocs_hooks", HOOKS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hooks():
    return _load_hooks()


def _render(hooks, html: str, page_url: str = "", docs_dir: Path = REPO_ROOT / "docs") -> str:
    return hooks.on_page_content(
        html, page=SimpleNamespace(url=page_url), config={"docs_dir": str(docs_dir)}, files=None
    )


def _png_bytes(width: int, height: int = 1) -> bytes:
    """Return a PNG signature plus an IHDR chunk header — all the hook reads."""
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + bytes(10)


def _webp_bytes(chunk: bytes, width: int, height: int = 1) -> bytes:
    """Return a RIFF/WEBP header with the first chunk of the given kind carrying the width."""
    if chunk == b"VP8 ":
        payload = bytes(3) + b"\x9d\x01\x2a" + struct.pack("<HH", width, height)
    elif chunk == b"VP8L":
        payload = b"\x2f" + struct.pack("<I", (width - 1) | ((height - 1) << 14))
    else:
        payload = bytes(4) + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    payload += bytes(20)
    riff_size = struct.pack("<I", 4 + 8 + len(payload))
    return b"RIFF" + riff_size + b"WEBP" + chunk + struct.pack("<I", len(payload)) + payload


# =================================================================================================
#  Rewriting
# =================================================================================================
@pytest.mark.parametrize(
    "page_url, expected",
    [
        ("", 'src="images/hero_light.svg"'),
        ("getting_started/", 'src="../images/hero_light.svg"'),
        ("benchmarks/third_party/head_to_head/tier1/", 'src="../../../../images/hero_light.svg"'),
    ],
)
def test_paths_are_re_anchored_onto_the_including_page(hooks, page_url, expected):
    # --- arrange ----------------------
    html = '<img src="docs/images/hero_light.svg" alt="x">'

    # --- act --------------------------
    rendered = _render(hooks, html, page_url)

    # --- assert -----------------------
    assert expected in rendered


def test_srcset_is_rewritten_too(hooks):
    """The dark variant of the hero rides on `srcset`, which a `src`-only rule would miss."""
    # --- arrange ----------------------
    html = '<source media="(prefers-color-scheme: dark)" srcset="docs/images/hero_dark.svg">'

    # --- act --------------------------
    rendered = _render(hooks, html)

    # --- assert -----------------------
    assert 'srcset="images/hero_dark.svg"' in rendered


@pytest.mark.parametrize(
    "html",
    [
        '<img src="images/splash.webp">',
        '<img src="https://raw.githubusercontent.com/bertpl/max-div/v0.8.3/images/splash.webp">',
        '<a href="docs/images/hero_light.svg">link</a>',
    ],
)
def test_unrelated_paths_are_left_alone(hooks, html):
    # --- act / assert -----------------
    assert _render(hooks, html) == html


# =================================================================================================
#  The README the hook exists for
# =================================================================================================
def test_every_readme_image_resolves_on_the_docs_site(hooks):
    """Each path the hook rewrites must name a file that actually ships in the docs tree."""
    # --- arrange ----------------------
    readme = README.read_text(encoding="utf-8")

    # --- act --------------------------
    referenced = [path for _attribute, path in hooks.DOCS_RELATIVE_ATTR.findall(readme)]
    missing = [path for path in referenced if not (REPO_ROOT / "docs" / path).exists()]

    # --- assert -----------------------
    assert referenced, "README no longer carries docs-relative image paths — is this hook still needed?"
    assert not missing, f"README references files absent from docs/: {missing}"


def test_the_rendered_readme_keeps_no_docs_relative_paths(hooks):
    # --- act --------------------------
    rendered = _render(hooks, README.read_text(encoding="utf-8"))

    # --- assert -----------------------
    assert not hooks.DOCS_RELATIVE_ATTR.findall(rendered)


# =================================================================================================
#  Figure display width
# =================================================================================================
@pytest.mark.parametrize(
    "name, content, expected",
    [
        ("lossy.webp", _webp_bytes(b"VP8 ", 1354), 1354),
        ("lossless.webp", _webp_bytes(b"VP8L", 2000), 2000),
        ("extended.webp", _webp_bytes(b"VP8X", 7268), 7268),
        ("plain.png", _png_bytes(640), 640),
        ("other.webp", b"not an image at all", None),
    ],
)
def test_pixel_width_is_read_from_the_header(hooks, tmp_path, name, content, expected):
    # --- arrange ----------------------
    path = tmp_path / name
    path.write_bytes(content)

    # --- act / assert -----------------
    assert hooks.image_pixel_width(path) == expected


def test_figure_dpi_comes_from_the_style_sheet(hooks):
    """The hook and every figure generator must agree on the one dpi, so it is read from the style sheet."""
    # --- act / assert -----------------
    assert hooks.figure_dpi() == 300.0


def test_a_raster_figure_is_sized_and_centered(hooks, tmp_path):
    """An 8-inch figure at 300 dpi is 2400 px wide and is shown at 8 in x 96 px/in = 768 CSS px.

    The `src` is URL-relative (MkDocs rewrote it before the hook runs), so `../images/` from the
    page directory `guides/page/` names `guides/images/` in the docs tree.
    """
    # --- arrange ----------------------
    (tmp_path / "guides" / "images").mkdir(parents=True)
    (tmp_path / "guides" / "images" / "chart.webp").write_bytes(_webp_bytes(b"VP8 ", 2400))
    html = '<img alt="a chart" src="../images/chart.webp" />'

    # --- act --------------------------
    rendered = _render(hooks, html, "guides/page/", tmp_path)

    # --- assert -----------------------
    assert rendered == '<img alt="a chart" src="../images/chart.webp" class="figure" width="768" />'


def test_an_existing_class_is_extended(hooks, tmp_path):
    # --- arrange ----------------------
    (tmp_path / "guides" / "page").mkdir(parents=True)
    (tmp_path / "guides" / "page" / "chart.png").write_bytes(_png_bytes(300))
    html = '<img class="center" src="chart.png">'

    # --- act --------------------------
    rendered = _render(hooks, html, "guides/page/", tmp_path)

    # --- assert -----------------------
    assert rendered == '<img class="center figure" src="chart.png" width="96">'


@pytest.mark.parametrize(
    "html",
    [
        '<img src="chart.webp" width="32%">',
        '<img src="chart.webp" style="max-width: 60%">',
        '<img src="https://example.org/chart.webp">',
        '<img src="hero.svg">',
        '<img src="missing.webp">',
    ],
)
def test_hand_sized_remote_vector_and_missing_images_are_left_alone(hooks, tmp_path, html):
    # --- arrange ----------------------
    (tmp_path / "guides" / "page").mkdir(parents=True)
    (tmp_path / "guides" / "page" / "chart.webp").write_bytes(_webp_bytes(b"VP8 ", 2400))
    (tmp_path / "guides" / "page" / "hero.svg").write_text("<svg/>")

    # --- act / assert -----------------
    assert _render(hooks, html, "guides/page/", tmp_path) == html


def test_the_home_page_is_not_sized(hooks, tmp_path):
    """The README's images are styled for GitHub and stay as they are."""
    # --- arrange ----------------------
    (tmp_path / "chart.webp").write_bytes(_webp_bytes(b"VP8 ", 2400))
    html = '<img src="chart.webp">'

    # --- act / assert -----------------
    assert _render(hooks, html, "", tmp_path) == html
