import numpy as np
import tenseal as ts

from thex import logger


def ReLU(X, context_manager):
    @context_manager.depth_updater()
    def relu(X):
        o = np.zeros_like(X)
        return np.maximum(X, o)
    
    # decrypt X
    X = X.decrypt()
    # ReLU
    X = relu(X)
    # encrypt X
    return ts.ckks_vector(context_manager.context, X)
