import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

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
    def setUp(self):
        self.query = torch.randn(2, 3, 4)
        self.key = torch.randn(2, 3, 4)
        self.value = torch.randn(2, 3, 4)
        self.mask = torch.ones(2, 3, 3)
        self.mask[:, 1, 1] = 0
        self.mask[:, 2, 2] = 0
        self.mask[:, 2, 1] = 0
        self.mask[:, 1, 2] = 0
        logger(f"query: {self.query}")
        logger(f"key: {self.key}")
        logger(f"value: {self.value}")
        logger(f"mask: {self.mask}")

        self.attn = Attention()
        self.encattn = EncAttention()
    def test_attention(self):
        output, attn = self.attn(
            self.query, 
            self.key, 
            self.value, 
            mask=self.mask, dropout=None)

        logger(f"output: {output}")

        self.assertEqual(output.shape, (2, 3, 4))
        self.assertEqual(attn.shape, (2, 3, 3))
        self.assertEqual(attn[0, 1, 1], 0)
        self.assertEqual(attn[0, 2, 2], 0)
        self.assertEqual(attn[0, 2, 1], 0)
        self.assertEqual(attn[0, 1, 2], 0)

    def test_enc_attention(self):

        output, attn = self.attn(
            self.query, 
            self.key, 
            self.value, 
            mask=self.mask, 
            dropout=None)
        logger(f"output: {output}")

        enc_query = cxt_man.encrypt(self.query)
        enc_key = cxt_man.encrypt(self.key)
        enc_value = cxt_man.encrypt(self.value)
        enc_output, enc_attn = self.encattn(
            enc_query,
            enc_key,
            enc_value,
            mask=self.mask,
            dropout=None)
        dec_output = cxt_man.decrypt(enc_output)
        logger(f"enc_output: {dec_output}")


if __name__ == '__main__':
    unittest.main()
