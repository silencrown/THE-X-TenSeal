import unittest
import torch

import test_helper
from thex import cxt_man
from thex import logger

from thex.xnn.softmax import (
    SoftmaxApprox,
    EncSoftmax
)
from thex.convert.softmax_approx import (
    SoftmaxApproxTrainer,
)

class TestSoftmax(unittest.TestCase):
    def softmax(self, x):
        return torch.nn.Softmax(dim=-1)(x)
    
    def test_softmax_approx(self):
        # generate input tensor
        input_tensor = torch.randn(1, 2, 128, 128)
        # generate label
        label = self.softmax(input_tensor)
        # generate model (default: use pretrained model)
        model = SoftmaxApprox()
        # get result
        result = model(input_tensor)
        # compare result with label
        self.assertTrue(torch.allclose(result, label, atol=1e-3))




if __name__ == '__main__':
    unittest.main()
