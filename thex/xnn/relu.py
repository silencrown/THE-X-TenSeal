import numpy as np
import torch
import tenseal as ts
import torch.nn as nn

from thex import cxt_man
from thex import logger
from thex import utils


def ReLU(self, input_tensor):
    x = torch.relu(input_tensor)
    return x

@cxt_man.depth_renew
def EncReLU(self, enc_x):
    """
    Encrypted ReLU
    """
    # decrypt
    x = np.array(enc_x.decrypt().tolist())
    # relu
    x = np.maximum(0, x)
    # encrypt
    enc_x = cxt_man.encrypt(x)
    # TODO: unittest this
    return enc_x
