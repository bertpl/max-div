#!/bin/python3
# SYNTAX:
#  python3 update_readme_urls.py <python_versions_file> <pyproject_toml_file> <readme_file> <build_type>
import re
import sys
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


def update_readme(readme_file: Path, python_versions: list[str], package_version: str, build_type: str):
    # --- read file ---------------------------------------
    readme_lines = readme_file.read_text().splitlines()

    # --- update python versions badge --------------------
    url_python_versions = "%20%7C%20".join(python_versions)  # ' | ' is the separator in the shields.io URL
    badge_link = f"![shields.io-python-versions](https://img.shields.io/badge/python-{url_python_versions}-blue)"
    for i, line in enumerate(readme_lines):
        if line.startswith("![shields.io-python-versions]"):
            if line != badge_link:
                print(f"Updating line {i + 1} in README.md:")
                print(f"  OLD: {line}")
                print(f"  NEW: {badge_link}")
            readme_lines[i] = badge_link
            break

    # --- update links to gh_pages ------------------------
    relative_version_path = f"{build_type}/v{package_version}"
    pattern = r"(release|develop)/v\d+\.\d+\.\d+"
    for i, line in enumerate(readme_lines):
        new_line = re.sub(pattern, relative_version_path, line)
        if new_line != line:
            print(f"Updating line {i + 1} in README.md:")
            print(f"  OLD: {line}")
            print(f"  NEW: {new_line}")
        readme_lines[i] = re.sub(pattern, relative_version_path, line)

    # --- save update lines -------------------------------
    readme_file.write_text("\n".join(readme_lines))


# -------------------------------------------------------------------------
#  Main entrypoint
# -------------------------------------------------------------------------
def update_readme_urls():
    # --- argument handling -------------------------------
    args = sys.argv[1:]
    if len(args) != 4:
        print(f"expected 4 arguments, got {len(args)}.")
        print(f"Syntax:")
        print(f"  update_readme_urls.py <python_versions_file> <pyproject_toml_file> <readme_file> <build_type>")
        exit(1)
    else:
        python_versions_file = Path(args[0])
        pyproject_toml_file = Path(args[1])
        readme_file = Path(args[2])
        build_type = args[3]
        if build_type not in ["release", "develop"]:
            print(f"4th argument must be 'release' or 'develop', got '{build_type}'.")
            exit(1)

    # --- read data ---------------------------------------
    package_version = get_package_version(pyproject_toml_file)
    python_versions = get_python_versions(python_versions_file)

    # --- update readme -----------------------------------
    update_readme(readme_file, python_versions, package_version, build_type)


if __name__ == "__main__":
    print(f"Updating README badge URLs...")
    update_readme_urls()
