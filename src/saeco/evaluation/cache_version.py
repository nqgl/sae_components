from functools import wraps


def cache_version(v):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        setattr(wrapper, "_version", v)
        return wrapper

    return decorator
