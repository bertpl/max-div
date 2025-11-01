from typing import Callable


def dummy_decorator(*args, **kwargs):
    # dummy decorator that does nothing and can be used with or without arguments
    if len(args) == 1 and isinstance(args[0], Callable):
        # decorator used without arguments
        return args[0]
    else:
        # decorator used with arguments
        def decorator(func):
            return func

        return decorator
