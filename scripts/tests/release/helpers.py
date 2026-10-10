"""This module holds the helpers shared by the tests of the release steps."""

import importlib
import sys
from pathlib import Path
from types import ModuleType

RELEASE_DIR = Path(__file__).resolve().parents[2] / "release"


def load_release_module(name: str) -> ModuleType:
    """Import module `name` of the scripts/release/ folder, the way `python scripts/release` resolves it.

    The folder runs by its path, so its modules import each other by module name alone; this function puts
    the folder on `sys.path`, so those imports resolve in the tests too.
    """
    release_dir = str(RELEASE_DIR)
    if release_dir not in sys.path:
        sys.path.insert(0, release_dir)
    return importlib.import_module(name)
