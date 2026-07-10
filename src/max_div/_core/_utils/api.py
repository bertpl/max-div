def patch_modules(globs: dict, names: list[str], module: str) -> None:
    """Patch ``__module__`` on re-exported names so they appear to belong to *module*.

    For bound methods (e.g. classmethods exported as module-level names) the
    underlying ``__func__`` is patched instead, since the method wrapper itself
    does not support attribute assignment.

    Args:
        globs:  The ``globals()`` dict of the re-exporting module.
        names:  The names to patch (typically the module's ``__all__``).
        module: The desired ``__module__`` string (typically ``__name__``).
    """
    for name in names:
        obj = globs[name]
        target = getattr(obj, "__func__", obj)
        if hasattr(target, "__module__"):
            target.__module__ = module
