from abc import ABC, abstractmethod
import tenseal as ts


class AbstractEncryptedLayer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
