import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import EncLinear
from .relu import EncReLU
from .Module import FHELayer


class GELU(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created.
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_1 = nn.Linear(d_model, d_ff)
        self.layer_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.layer_2(self.dropout(self.activation(self.layer_1(x))))

class ApproxPositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(ApproxPositionwiseFeedForward, self).__init__()

        self.activation = F.relu()

    def forward()
    
class EncPositionwiseFeedForward(FHELayer):
    "Implements FFN equation."

    def __init__(self, torch_nn):
        # TODO: may need to add dropout
        super(EncPositionwiseFeedForward, self).__init__()
        self.layer_1 = EncLinear(torch_nn.layer_1)
        self.layer_2 = EncLinear(torch_nn.layer_2)
        self.activation = EncReLU()

    def forward(self, enc_x):
        enc_x = self.layer_1(enc_x)
        return self.layer_2(self.activation(enc_x))
