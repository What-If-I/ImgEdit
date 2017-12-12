from functools import wraps
import importlib


def compare_with_method(fn_name):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            f = getattr(importlib.import_module('cv2'), fn_name)
            args += extra_args.get(fn_name, ())
            kwargs.update(extra_kwargs.get(fn_name, {}))
            return f(*args, **kwargs)

        return wrapper

    return decorator


extra_args = {
    "Sobel": (6, 2, 0,),
}

extra_kwargs = {
    "Soble": {'ksize': 3},
}
