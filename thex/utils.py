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
    if isinstance(encdata, ts.ckks_vector):
        return "vector"
    elif isinstance(encdata, ts.ckks_tensor):
        return "tensor"
    else:
        raise ValueError("Invalid input array")