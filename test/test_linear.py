import numpy as np
import torch
import unittest
import tenseal as ts

import test_helper
from thex import logger
from thex.ContextManager import ContextManager
from thex.xnn.linear import (
    EncLinear, 
    Linear
)


class TestLinear(unittest.TestCase):
    def setUp(self):
        self.context_manager = ContextManager()

    def test_Linear(self):
        # create a random input tensor
        tensor = torch.randn(1, 8, requires_grad=False)
        enc_tensor = ts.ckks_tensor(self.context_manager.context, tensor.tolist())
        # logger(f"tensor: {tensor[0]}")

        # torch model inference
        torch_model = Linear(8, 8)
        result = torch_model(tensor).tolist()
        logger(f"result: {result}")

        # enc model inference
        enc_model = EncLinear(torch_model)
        enc_result = np.array(enc_model(enc_tensor).decrypt().tolist())
        logger(f"enc_result: {enc_result}")
        # logger.log_system_info()
        
        # decrypt and check the values
        np.testing.assert_array_almost_equal(enc_result, 
                                             result, 
                                             decimal=4)


if __name__ == '__main__':
    unittest.main()
