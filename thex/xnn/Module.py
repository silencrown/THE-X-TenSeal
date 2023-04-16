import torch
import tenseal as ts
from abc import ABC, abstractmethod


class EncModuleList:
    def __init__(self, layers=None):
        self.layers = []

        if layers is not None:
            if not all(isinstance(layer, FHELayer) for layer in layers):
                raise TypeError("All elements in the list must be instances of FHELayer")
            self.layers.extend(layers)

    def append(self, layer):
        if not isinstance(layer, FHELayer):
            raise TypeError("The element to be appended must be an instance of FHELayer")
        self.layers.append(layer)

    def __getitem__(self, index):
        return self.layers[index]

    def __setitem__(self, index, value):
        if not isinstance(value, FHELayer):
            raise TypeError("The element to be assigned must be an instance of FHELayer")
        self.layers[index] = value

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

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
