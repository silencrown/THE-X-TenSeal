import torch
import tenseal as ts
from abc import ABC, abstractmethod


class FHELayer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Module(torch.nn.Module):
    """Wrapper class for torch.nn.Module to add FHE support."""

    def __init__(self, encryption_context=None):
        super(Module, self).__init__()
        self.is_encrypted = False
        self.encryption_context = encryption_context

    def encrypt(self, encryption_context=None):
        """Switches the module to FHE mode."""
        self.is_encrypted = True
        if not encryption_context:
            self.encryption_context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[40, 40, 40, 40])

    def decrypt(self):
        """Switches the module to plain mode."""
        self.is_encrypted = False
        self.encryption_context = None

    def forward(self, *inputs):
        """Abstract method that should be overridden by all subclasses."""
        raise NotImplementedError
