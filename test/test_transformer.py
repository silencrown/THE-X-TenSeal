import unittest

import numpy as np
import torch

import test_helper
from thex import (
    logger,
    cxt_man,
)
from thex.xnn.attention import (
    Attention, 
    EncAttention,
    MultiHeadedAttention, 
    EncMultiHeadedAttention,
)


class TestAttention(unittest.TestCase):

    def setUp(self) -> None:
        # Notice: cannot use batch
        # self.batch_size = 2
        self.seq_len = 4
        self.d_model = 16
        self.num_heads = 4
        self.batch_size = 1

        self.query = torch.randn(self.seq_len, self.d_model)
        self.key = torch.randn(self.seq_len, self.d_model)
        self.value = torch.randn(self.seq_len, self.d_model)   


if __name__ == '__main__':
    unittest.main()
