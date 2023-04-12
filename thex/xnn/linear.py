import torch
import torch.nn as nn
import tenseal as ts

from thex.xnn.Module import FHELayer
from thex import logger


class EncLinear(FHELayer):
    def __init__(self, torch_nn):
        super(EncLinear, self).__init__()
        if isinstance(torch_nn, nn.Linear):
            self.init_from_torch_linear(torch_nn)
        elif isinstance(torch_nn, nn.Module):
            self.init_from_torch_module(torch_nn)
        else:
            raise ValueError("torch_nn must be nn.Linear or nn.Module contains a nn.Linear layer named `fc`")
    
    def init_from_torch_module(self, torch_nn):
        """
        Args:
        - torch_nn: nn.Module

        Notice:
        - torch_nn most have a nn.Linear layer named `fc`
        """
        self.fc_weight = torch_nn.fc.weight.T.data.tolist()
        self.fc_bias = torch_nn.fc.bias.data.tolist()

    def init_from_torch_linear(self, torch_nn):
        """
        Args:
        - torch_nn: nn.Linear
        """
        self.fc_weight = torch_nn.weight.T.data.tolist()
        self.fc_bias = torch_nn.bias.data.tolist()

    def forward(self, enc_x):
        enc_x = enc_x.mm(self.fc_weight) + self.fc_bias
        return enc_x

class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input_tensor):
        x = self.fc(input_tensor)
        return x