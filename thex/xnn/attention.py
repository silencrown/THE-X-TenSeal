import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tenseal as ts

from thex import (
    logger,
    cxt_man, 
    utils,
    )
from .Module import FHELayer
from .softmax import EncSoftmax


def masked_fill(mask, value):
    """
    Mask a tensor with a value
    Args:
    - mask: a boolean tensor
    - value: a float number
    """
    return mask * value + (1 - mask) * 1e-9

def transpose(matrix):
    """
    Transpose a 2D list
    Args:
    - matrix: a 2D list
    """
    return list(map(list, zip(*matrix)))

class Attention(nn.Module):
    """
    Torch Class of Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    
class EncAttention(FHELayer):
    def __init__(self):
        super(EncAttention, self).__init__()
        self.softmax = EncSoftmax()
    """
    Enc Torch Class of Compute' Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = query.mm(key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = masked_fill(scores, mask == 0, -1e9)

        p_attn = self.softmax(scores, dim=-1)

        # if dropout is not None:
        #     p_attn = dropout(p_attn)

        return p_attn.mm(value), p_attn

class EncMultiHeadedAttention(FHELayer):
    """
    Enc Torch Class of Multi-Headed Attention
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(EncMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h

    def forward():
        pass
    
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)