import tenseal as ts
import torch
import numpy as np

from xnn.Module import FHELayer

class OriginLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(OriginLayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
class ApproxLayerNorm(torch.nn.Module):
    def __init__(self, input_size):
        super(ApproxLayerNorm, self).__init__()
        self.fc = torch.nn.Linear(input_size, input_size)
    
    def forward(self, x):
        return self.fc(x)
    
class enc_layernorm(FHELayer):
    def __init__(self, torch_nn):
        super(enc_layernorm, self).__init__()
        self.fc_weight = torch_nn.fc.weight.T.data.tolist()
        self.fc_bias = torch_nn.fc.bias.data.tolist()

    def forward(self, enc_x):
        """
        y = x ◦ γ + β
        """
        enc_x = enc_x.mm(self.fc_weight) + self.fc_bias
        return enc_x
    