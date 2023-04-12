import numpy as np
import unittest
import torch
import tenseal as ts

import test_helper
from thex.xnn.ffn import EncPositionwiseFeedForward
from thex.xnn.ffn import PositionwiseFeedForward
from thex import logger
from thex import cxt_man


class TestFeedForward(unittest.TestCase):
    
    def get_torch_model(self, input_size, hidden_size, dropout=0.0):
        return PositionwiseFeedForward(input_size, hidden_size)
    
    def get_thex_model(self, torch_model):
        return EncPositionwiseFeedForward(torch_model)

    def test_ffn(self):
        logger.log_cxt_info(cxt_man)

        # generate model
        torch_model = self.get_torch_model(10, 10)
        thex_model = self.get_thex_model(torch_model)

        # generate input tensor
        x = torch.randn(1, 10, requires_grad=False)
        enc_x = cxt_man.encrypt(x.tolist())

        # model inference
        torch_y = torch_model(x).tolist()
        enc_y = thex_model(enc_x)

        # decrypt and compare
        dec_y = cxt_man.decrypt(enc_y)
        logger(f"torch_y: {torch_y}")
        logger(f"enc_y: {dec_y}")
        np.testing.assert_array_almost_equal(torch_y, dec_y, decimal=1)


if "__main__" == __name__:
    unittest.main()