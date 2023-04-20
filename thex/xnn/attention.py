import math, copy

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
from .Module import FHELayer, EncModuleList
from .linear import EncLinear
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

def clones(module, N=6):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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
        
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = Attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
class EncAttention(FHELayer):
    def __init__(self):
        super(EncAttention, self).__init__()
        self.softmax = EncSoftmax()
    """
    Enc Torch Class of Compute' Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = query.mm(key.transpose(-2, -1)) 
        scores = scores * (1.0 / math.sqrt(query.size(-1))) # use scalar multiplication

        if mask is not None:
            scores = masked_fill(scores, mask == 0, -1e9)

        p_attn = self.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return p_attn.mm(value), p_attn

class EncMultiHeadedAttention(FHELayer):
    """
    Enc Torch Class of Multi-Headed Attention
    """
    def __init__(self, h, d_model):
        super(EncMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = EncModuleList([EncLinear(d_model, d_model) for _ in range(3)])
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, self.attn = EncAttention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linear_layers[-1](x)
    
