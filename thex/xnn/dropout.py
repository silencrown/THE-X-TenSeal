import random
from typing import List

import torch

from thex import logger, cxt_man
from .Module import FHELayer


class Dropout(torch.nn.Module):
    """Implements a dropout layer that randomly sets each element in the input to zero with probability `p`.
    
    Args:
        p: A float between 0 and 1, representing the probability of setting each element to zero.
    """
    def __init__(self, p: float):
        super().__init__()
        self.p = p
    
    def forward(self, x: List[float]) -> List[float]:
        """Implements a dropout layer that randomly sets each element in the input to zero with probability `p`.
        
        Args:
            x: A list of numbers, representing the input.
        
        Returns:
            A list of numbers with the same length as `x`, representing the output of the dropout layer.
        """
        return self.dropout(x, self.p)

    @staticmethod
    def dropout(x: List[float], p: float) -> List[float]:
        """Implements a dropout function that randomly sets each element in the input `x` to zero with probability `p`.
        
        Args:
            x: A list of numbers, representing the input.
            p: A float between 0 and 1, representing the probability of setting each element to zero.
        
        Returns:
            A list of numbers with the same length as `x`, representing the output of the dropout function.
        
        Raises:
            TypeError: If `x` is not a list.
            ValueError: If `p` is not a float between 0 and 1.
        """
        if not isinstance(x, list):
            raise TypeError("Input 'x' must be a list.")
        if not isinstance(p, float) or p < 0 or p > 1:
            raise ValueError("Probability 'p' must be a float between 0 and 1.")
        
        mask = [1 if random.random() > p else 0 for _ in range(len(x))]
        return [x[i] * mask[i] for i in range(len(x))]


class EncDropout(FHELayer):
    """Implements a dropout layer that randomly sets each encrypted element in the input to zero with probability `p`."""

    def __init__(self, torch_nn):
        super().__init__()
        self.p = torch_nn.p

    def forward(self, enc_x):
        """Implements a dropout layer that randomly sets each element in the input to zero with probability `p`.
        
        Args:
            enc_x: An encrypted tensor with shape (batch_size, seq_length, hidden_size), the input to be normalized.
        
        Returns:
            An encrypted tensor with the same shape as `enc_x`, the normalized output.
        """
        return self.enc_dropout(enc_x, self.p)

    @staticmethod
    def enc_dropout(enc_x, p):
        """Implements a dropout function that randomly sets each element in the input `x` to zero with probability `p`.
        
        Args:
            enc_x: An encrypted tensor with shape (batch_size, seq_length, hidden_size), the input to be normalized.
            p: A float between 0 and 1, representing the probability of setting each element to zero.
        
        Returns:
            An encrypted tensor with the same shape as `enc_x`, the normalized output.
        """
        mask = [1 if random.random() > p else 0 for _ in range(len(enc_x))]
        return enc_x * mask