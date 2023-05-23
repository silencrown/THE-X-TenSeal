import math, copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thex import (
    logger,
    cxt_man, 
    utils,
    )
from .Module import FHELayer, EncModuleList
from .linear import EncLinear
from .softmax import EncSoftmax, EncSoftmaxApprox
from .transpose import transpose


def masked_fill(mask, value):
    """
    Mask a tensor with a value
    Args:
    - mask: a boolean tensor
    - value: a float number
    """
    return mask * value + (1 - mask) * 1e-9

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

class ApproxAttention(nn.Module):
    def __init__(self, softmax=F.softmax):
        super(ApproxAttention, self).__init__()
        self.softmax = softmax
    
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = masked_fill(scores, mask == 0, -1e9)

        p_attn = self.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class EncAttention(FHELayer):
    def __init__(self, d, softmax=EncSoftmax()):
        super(EncAttention, self).__init__()
        self.softmax = softmax
        self.d = d
    """
    Enc Torch Class of Compute' Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        key = transpose(key, [-2, -1])
        scores = query.mm(key)
        repi_sqrt_d = 1.0 / math.sqrt(self.d)
        scores = scores * repi_sqrt_d

        if mask is not None:
            scores = masked_fill(scores, mask == 0, -1e9)

        p_attn = self.softmax(scores, dim=-1) # attention prob

        if dropout is not None:
            p_attn = dropout(p_attn)

        return p_attn.mm(value), p_attn

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        Args:
        - h: number of heads
        - d_model: dimension of model
        - dropout: dropout rate
        """

        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
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
    
    def enc_forward(self, query, key, value, mask=None):
        """
        forward function without 3Dtensor.matmul

        Method:
        - The matrix operations in the original class have all been changed to operations on independent vectors.
            1. Handle query, key, value separately on each head.
            2. Then concatenate the obtained results.
            3. Finally, perform a linear transformation.

        Notice:
        - Assumes that the input of enc_forward is 2D tensor (seq_len, d_model).
        - Lost the parallel advantage of multi-head attention.
        """

        query, key, value = [l(x) for l, x in zip(self.linear_layers, (query, key, value))]
        logger.debug(f"query: {query.shape}, key: {key.shape}, value: {value.shape}")
        output = torch.zeros_like(query)
        logger.debug(f"output: {output.shape}")
        for i in range(self.h):
            h_query = query[:, i*self.d_k:(i+1)*self.d_k]
            h_key = key[:, i*self.d_k:(i+1)*self.d_k]
            h_value = value[:, i*self.d_k:(i+1)*self.d_k]
            logger.debug(f"h_query: {h_query.shape}, h_key: {h_key.shape}, h_value: {h_value.shape}")

            h_output, _ = self.attention(h_query, h_key, h_value, mask=mask, dropout=self.dropout)
            output[:, i*self.d_k:(i+1)*self.d_k] = h_output
            logger.debug(f"h_output: {h_output.shape}")

        return self.output_linear(output)
    
    
class ApproxMultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):

        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = ApproxAttention()
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
    

class EncMultiHeadedAttention(FHELayer):
    """
    Enc Class of Multi-Headed Attention
    """

    def __init__(self, h, d_model, torch_nn, dropout=None):
        super(EncMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.d_k = self.d_model // h
        self.h = h
        self.Q_linear = EncLinear(torch_nn.linear_layers[0])
        self.K_linear = EncLinear(torch_nn.linear_layers[1])
        self.V_linear = EncLinear(torch_nn.linear_layers[2])
        self.output_linear = EncLinear(torch_nn.output_linear)
        self.attention = EncAttention(d=self.d_k)
        self.dropout = dropout
        
    def forward(self, query, key, value, mask=None):
        """
        forward function without 3Dtensor.matmul

        Method:
        - The matrix operations in the original class have all been changed to operations on independent vectors.
            1. Handle query, key, value separately on each head.
            2. Then concatenate the obtained results.
            3. Finally, perform a linear transformation.

        Notice:
        - Assumes that the input of enc_forward is 2D tensor (seq_len, d_model).
        - Lost the parallel advantage of multi-head attention.
        """

        # query, key, value = [l(x) for l, x in zip(self.linear_layers, (query, key, value))]
        query = self.Q_linear(query)
        key = self.K_linear(key)
        value = self.V_linear(value)
        logger.debug(f"query: {query.shape}, key: {key.shape}, value: {value.shape}")

        outputs = []
        for i in range(self.h):
            h_query = query[:, i*self.d_k:(i+1)*self.d_k]
            h_key = key[:, i*self.d_k:(i+1)*self.d_k]
            h_value = value[:, i*self.d_k:(i+1)*self.d_k]
            logger.debug(f"h_query: {h_query.shape}, h_key: {h_key.shape}, h_value: {h_value.shape}")

            h_output, _ = self.attention(h_query, h_key, h_value, mask=mask, dropout=self.dropout)
            outputs.append(h_output)
            logger.debug(f"h_output: {h_output.shape}")
        
        output = self.combine_outputs(outputs, query.shape)

        return self.output_linear(output) 
    
    def combine_outputs(self, outputs, shape):
        # TODO: Is there a better way to combine each head's output?
        output = torch.zeros(*shape)

        for current_output, i in zip(outputs, range(self.h)):
            current_output = torch.tensor(cxt_man.decrypt(current_output))
            output[:, i*self.d_k:(i+1)*self.d_k] = current_output
        return cxt_man.encrypt(output)
