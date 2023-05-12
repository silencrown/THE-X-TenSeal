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

        self.query = torch.randn(self.seq_len, self.d_model)
        self.key = torch.randn(self.seq_len, self.d_model)
        self.value = torch.randn(self.seq_len, self.d_model)   

    def _test_attention(self):
        attn = Attention()
        output, attn = attn(self.query, self.key, self.value)
        # logger(f"output: {output}")
        return output

    def test_enc_attention(self):

        expected = self._test_attention().numpy()

        enc_query = cxt_man.encrypt(self.query.tolist())
        enc_key = cxt_man.encrypt(self.key.tolist())
        enc_value = cxt_man.encrypt(self.value.tolist())
        enc_attn_layer = EncAttention(d=self.key.size(-1))
        enc_output, enc_attn = enc_attn_layer(
            enc_query,
            enc_key,
            enc_value)
        
        dec_output = np.array(cxt_man.decrypt(enc_output))
        logger(f"test_enc_output: {dec_output}")
        logger(f"expected: {expected}")
        np.testing.assert_array_almost_equal(expected, dec_output, decimal=1)
        


if __name__ == '__main__':
    unittest.main()
