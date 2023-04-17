import tenseal as ts
import torch
import numpy as np

from .Module import FHELayer

class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
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
        """Initialize an ApproxLayerNorm module.
        
        Args:
            input_size: An integer, the number of features in the input.
        """
        super(ApproxLayerNorm, self).__init__()
        self.fc = torch.nn.Linear(input_size, input_size)
    
    def forward(self, x):
        """Apply approximate Layer Normalization to the input tensor `x`.
        
        Args:
            x: A tensor with shape (batch_size, seq_length, input_size), the input to be normalized.
        
        Returns:
            A tensor with the same shape as `x`, the normalized output.
        """
        return self.fc(x)
    
class EncLayerNorm(FHELayer):
    # TODO: use Linear layer instead of fc
    def __init__(self, torch_nn):
        super(EncLayerNorm, self).__init__()
        self.fc_weight = torch_nn.fc.weight.T.data.tolist()
        self.fc_bias = torch_nn.fc.bias.data.tolist()

    def forward(self, enc_x):
        """Apply encrypted Layer Normalization to the encrypted tensor `enc_x`.
        
        Args:
            enc_x: An encrypted tensor with shape (batch_size, seq_length, hidden_size), the input to be normalized.
        
        Returns:
            An encrypted tensor with the same shape as `enc_x`, the normalized output.
        """
        # y = x ◦ γ + β
        enc_x = enc_x.mm(self.fc_weight) + self.fc_bias
        return enc_x
    