from typing import List
import tenseal as ts
import numpy as np

def ndarray_type(arr: np.ndarray) -> str:
    if arr.ndim == 1:
        return "vector"
    elif arr.ndim >= 2:
        return "tensor"
    else:
        raise ValueError("Invalid input array")

def encdata_type(encdata) -> str:
    """Return the type of the encrypted data."""

    if isinstance(encdata, ts.ckks_vector):
        return "vector"
    elif isinstance(encdata, ts.ckks_tensor):
        return "tensor"
    else:
        raise ValueError("Invalid input array")

def get_axes_perm(shape, transpose):
    """Return the permutation of the axes for the transpose operation."""

    indexes = list(range(len(shape)))
    transpose = [i if i >= 0 else len(shape) + i for i in transpose]
    indexes[transpose[0]], indexes[transpose[1]] = indexes[transpose[1]], indexes[transpose[0]]
    return indexes
    