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

    def _test_attention(self):
        attn = Attention()
        output, attn = attn(self.query, self.key, self.value)
        # logger(f"output: {output}")
        return output
    
    def _test_enc_attention(self):

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

        
    def _test_approx_forward(self):
        """test 2D multi-head attention forward function"""

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        enc_simul_x = x.squeeze(0)

        multi_attn = MultiHeadedAttention(
            h=self.num_heads,
            d_model=self.d_model)
        multi_attn.dropout = None # disable dropout for testing
        
        output = multi_attn(x, x, x).detach().numpy()
        enc_output = multi_attn.enc_forward(enc_simul_x, enc_simul_x, enc_simul_x).unsqueeze(0).detach().numpy()

        # logger(f"torch multi_attn output: {output}")
        # logger(f"torch enc_multi_attn output: {enc_output}")
        
        logger(f"torch multi_attn output size: {output.shape}")
        logger(f"torch enc_multi_attn output size: {enc_output.shape}")

        np.testing.assert_array_almost_equal(output, enc_output, decimal=1)

    def test_enc_multihead_atten(self):
        """test enc-input multi-head attention forward function"""

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        enc_x = cxt_man.encrypt(x.squeeze(0).tolist())
        
        multi_attn = MultiHeadedAttention(
            h=self.num_heads,
            d_model=self.d_model)
        multi_attn.dropout = None # disable dropout for testing
        expected_output = multi_attn(x, x, x).squeeze(0).detach().numpy()

        enc_multi_attn = EncMultiHeadedAttention(multi_attn.h, multi_attn.d_model, multi_attn)
        enc_output = enc_multi_attn(enc_x, enc_x, enc_x)
        dec_output = np.array(cxt_man.decrypt(enc_output))
        logger(f"torch multi_attn output: {expected_output}")
        logger(f"torch enc_multi_attn output: {dec_output}")

        np.testing.assert_array_almost_equal(expected_output, dec_output, decimal=1)


if __name__ == '__main__':
    unittest.main()
