"""
Performance test of THE-X linear layer
"""

import numpy as np
import torch
import unittest
import tenseal as ts

import test_helper
from thex import logger
from thex import cxt_man
from thex.xnn.linear import (
    EncLinear, 
    Linear
)


def _test_Linear(self, num_size):
    # create a random input tensor
    tensor = torch.randn(1, num_size, requires_grad=False)
    enc_tensor = cxt_man.encrypt(tensor)
    # logger(f"tensor: {tensor[0]}")

    # torch model inference
    torch_model = Linear(num_size, num_size)
    result = torch_model(tensor).tolist()
    logger(f"result: {result[0][0]}")

    # enc model inference
    enc_model = EncLinear(torch_model)
    enc_result = np.array(enc_model(enc_tensor).decrypt().tolist())
    logger(f"enc_result: {enc_result[0][0]}")

    
    # decrypt and check the values
    np.testing.assert_array_almost_equal(enc_result, 
                                            result, 
                                            decimal=4)