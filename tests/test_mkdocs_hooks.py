"""Guards for the documentation-site build hooks.

The README's image paths are repo-root-relative so that GitHub resolves them; the docs build
re-anchors them onto the rendered page. These tests pin that rewrite, since nothing else fails
loudly when it stops happening — a broken image is invisible to the docs build.
"""

import importlib.util
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


def _render(hooks, html: str, page_url: str = "") -> str:
    return hooks.on_page_content(html, page=SimpleNamespace(url=page_url), config=None, files=None)


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
