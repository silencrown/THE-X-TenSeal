import unittest
import numpy as np
import tenseal as ts
import torch

import test_helper
from thex import cxt_man
from thex import logger
from thex.xnn.relu import ReLU, EncReLU

class TestReLU(unittest.TestCase):

    @cxt_man.depth_refresher()
    def fun_refresher(self, x):
        return x
    
    @cxt_man.depth_limiter(depth_increment=1)
    def fun_limiter(self, x):
        return x
    
    def _test_wrapper(self):
        """Test the wrapper return value"""
        x = 10
        logger(f"x: {x}")
        limiter_result = self.fun_limiter(x)
        logger(f"limiter_result: {limiter_result}")
        refresher_result = self.fun_refresher(x)
        logger(f"refresher_result: {refresher_result}")

    def test_ReLU(self):
        # x = np.array([-1., 0., 1.])
        logger(cxt_man)
        x = np.arange(1, 129).reshape(1, 128)
        enc_x = cxt_man.encrypt(x)
        logger(f"enc_x: {enc_x}")

        result = ReLU(x)
        enc_result = EncReLU(enc_x)
        logger(f"enc_result: {enc_result}")
        
        logger(f"result: {result}")
        logger(f"enc_result: {cxt_man.decrypt(enc_result)}")
        np.testing.assert_array_almost_equal(result, cxt_man.decrypt(enc_result), decimal=2)


if __name__ == '__main__':
    unittest.main()