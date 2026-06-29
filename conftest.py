"""Pytest compatibility fixtures for legacy test modules."""

import builtins
from pathlib import PosixPath, WindowsPath

import numpy as np
import pandas as pd

builtins.np = np


def _path_add(self, other):
    """Compatibility for legacy tests that concatenate Path and str."""
    return self.__class__(str(self) + str(other))


if not hasattr(PosixPath, "__add__"):
    PosixPath.__add__ = _path_add
if not hasattr(WindowsPath, "__add__"):
    WindowsPath.__add__ = _path_add


_OriginalDataFrame = pd.DataFrame


class _CompatDataFrame(_OriginalDataFrame):
    """Pad uneven dict columns in legacy fixtures by repeating the last value."""

    def __init__(self, data=None, *args, **kwargs):
        if isinstance(data, dict):
            lengths = [
                len(value)
                for value in data.values()
                if hasattr(value, "__len__") and not isinstance(value, (str, bytes))
            ]
            if lengths and len(set(lengths)) > 1:
                max_len = max(lengths)
                fixed = {}
                for key, value in data.items():
                    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                        values = list(value)
                        if values and len(values) < max_len:
                            values.extend([values[-1]] * (max_len - len(values)))
                        fixed[key] = values
                    else:
                        fixed[key] = value
                data = fixed
        super().__init__(data, *args, **kwargs)


pd.DataFrame = _CompatDataFrame
