#!/bin/python3
# SYNTAX:
#  python3 update_readme_urls.py <python_versions_file> <pyproject_toml_file> <readme_file>
import sys
import textwrap
import tomllib
from pathlib import Path


# -------------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------------
def get_package_version(pyproject_toml_file: Path) -> str:
    with open(pyproject_toml_file, "rb") as f:
        pyproject = tomllib.load(f)
    return pyproject["project"]["version"]


def get_python_versions(python_versions_file: Path) -> list[str]:
    """Reads the python versions from .python-versions file."""
    with open(python_versions_file, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def render_template(python_versions: list[str], package_version: str) -> list[str]:
    url_python_versions = "%20%7C%20".join(python_versions)  # ' | ' is the separator in the shields.io URL

    return [
        line
        for line in textwrap.dedent(
            f"""
                ![shields.io-python-versions](https://img.shields.io/badge/python-{url_python_versions}-blue)
                ![genbadge-test-count](https://bertpl.github.io/max-div/version_artifacts/v{package_version}/badge-test-count.svg)
                ![genbadge-test-coverage](https://bertpl.github.io/max-div/version_artifacts/v{package_version}/badge-coverage.svg)
                ![max-div logo](https://bertpl.github.io/max-div/version_artifacts/v{package_version}/splash.webp)
            """
        ).splitlines()
        if line
    ]


def write_to_readme(readme_file: Path, url_lines: list[str]):
    # config
    start_line = "<!--START_SECTION:images-->"
    end_line = "<!--END_SECTION:images-->"

    # read file
    readme_lines = readme_file.read_text().splitlines()

    # update in-memory
    i_start_line = readme_lines.index(start_line)
    i_end_line = readme_lines.index(end_line)
    readme_lines = readme_lines[: i_start_line + 1] + url_lines + readme_lines[i_end_line:]

    # save back to file
    readme_file.write_text("\n".join(readme_lines))


# -------------------------------------------------------------------------
#  Main entrypoint
# -------------------------------------------------------------------------
def update_readme_urls():
    # --- argument handling -------------------------------
    args = sys.argv[1:]
    if len(args) != 3:
        print(f"expected 3 arguments, got {len(args)}.")
        print(f"Syntax:")
        print(f"  update_readme_urls.py <python_versions_file> <pyproject_toml_file> <readme_file>")
        exit(1)
    else:
        python_versions_file = Path(args[0])
        pyproject_toml_file = Path(args[1])
        readme_file = Path(args[2])

    # --- read data ---------------------------------------
    package_version = get_package_version(pyproject_toml_file)
    python_versions = get_python_versions(python_versions_file)

    # --- render template ---------------------------------
    url_lines = render_template(python_versions, package_version)

    # --- write to readme ---------------------------------
    write_to_readme(readme_file, url_lines)


if __name__ == "__main__":
    print(f"Updating README badge URLs...")
    update_readme_urls()
