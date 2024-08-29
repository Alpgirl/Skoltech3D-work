import warnings
from functools import wraps


class SimpleNamespace(dict):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        dict.__setitem__(self, key, value)

    def __setitem__(self, key, value):
        object.__setattr__(self, key, value)
        dict.__setitem__(self, key, value)

    def copy(self):
        return SimpleNamespace(**self)

    def copy_with(self, **kwargs):
        new = SimpleNamespace()
        for k, v in self.items():
            if k not in kwargs:
                new[k] = v
            else:
                if isinstance(v, SimpleNamespace):
                    new[k] = v.copy_with(**kwargs[k])
                else:
                    new[k] = kwargs[k]
        for k, v in kwargs.items():
            if k not in new:
                new[k] = v
        return new

    def to_dict(self):
        new = dict()
        for k, v in self.items():
            if isinstance(v, SimpleNamespace):
                new[k] = v.to_dict()
            else:
                new[k] = v
        return new


def ignore_warnings(msgs):
    def dec(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                for msg in msgs:
                    warnings.filterwarnings('ignore', message=msg)
                return f(*args, **kwargs)
        return wrapper
    return dec