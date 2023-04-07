import numpy as np
import torch
import tenseal as ts

from thex.xnn.Module import FHELayer
from thex import logger


class enc_ReLU(FHELayer):
    def __init__(self, context_manager):
        self.context_manager = context_manager

    def relu(X):
        o = np.zeros_like(X)
        return np.maximum(X, o)
    
    # decrypt X
    X = X.decrypt()
    # ReLU
    X = relu(X)
    # encrypt X
    return ts.ckks_vector(context_manager.context, X)

class enc_ReLU()
