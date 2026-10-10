"""This module holds the helpers shared by the tests of the maintainer tooling."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    """Import `scripts/<name>.py` as top-level module `name`, the way it runs as `python scripts/<name>.py`.

    A script imports another script by its bare name, and `scripts/` is not on `sys.path`, so load
    the scripts it imports first: each module is registered in `sys.modules` under `name` before its
    code runs, so a script loaded afterwards that runs `import <name>` gets that module.
    """
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_release_module(name: str) -> ModuleType:
    """Import module `name` of the scripts/release/ folder, the way `python scripts/release` resolves it.

    The folder runs by its path, so its modules import each other by bare name; this puts the folder
    on `sys.path`, so those imports resolve in the tests too.
    """
    release_dir = str(SCRIPTS_DIR / "release")
    if release_dir not in sys.path:
        sys.path.insert(0, release_dir)
    return importlib.import_module(name)
