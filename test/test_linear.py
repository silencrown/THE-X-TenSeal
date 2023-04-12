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

    def test_Linear_size(self):
        for num_size in range(1, 100):
            self._test_Linear(num_size)
            logger(f"input tensor size: {num_size} passed")

    def _test_Linear(self, num_size):
        logger.log_system_info()
        # create a random input tensor
        tensor = torch.randn(1, num_size, requires_grad=False)
        enc_tensor = ts.ckks_tensor(self.context_manager.context, tensor.tolist())
        # logger(f"tensor: {tensor[0]}")

        # torch model inference
        torch_model = Linear(num_size, num_size)
        result = torch_model(tensor).tolist()
        logger(f"result: {result[0][0]}")

        # enc model inference
        enc_model = EncLinear(torch_model)
        logger.log_system_info()
        enc_result = np.array(enc_model(enc_tensor).decrypt().tolist())
        logger(f"enc_result: {enc_result[0][0]}")

        
        # decrypt and check the values
        np.testing.assert_array_almost_equal(enc_result, 
                                             result, 
                                             decimal=4)

if __name__ == '__main__':
    unittest.main()
