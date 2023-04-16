import numpy as np

from thex import cxt_man
from thex import logger
from .Module import FHELayer


def ReLU(x):
    x = np.maximum(0, x)
    return x

@cxt_man.depth_refresher()
def enc_ReLU(enc_x):
    """
    Encrypted ReLU function
    """
    # decrypt
    x = np.array(cxt_man.decrypt(enc_x))
    logger(f"EncReLU input shape: {x.shape}")
    # relu
    x = np.maximum(0, x)
    # encrypt
    enc_x = cxt_man.encrypt(x)
    logger(f"EncReLU output shape: {enc_x.shape}")
    logger(f"EncReLU output: {enc_x}")
    return enc_x

class EncReLU(FHELayer):
    """
    Encrypted ReLU
    """
    def __init__(self):
        super(EncReLU, self).__init__()
        self.relu = enc_ReLU

    def forward(self, x):
        return self.relu(x)
