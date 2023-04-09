import numpy as np
import torch
import tenseal as ts
import torch.nn as nn

from thex.xnn.Module import FHELayer
from thex import logger


class enc_ReLU(FHELayer):
    def __init__(self, context_manager):
        self.context_manager = context_manager

    def relu(X):
        # TODO: implement relu
        pass

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        pass

    def forward(self, input_tensor):
        x = torch.relu(input_tensor)
        return x
