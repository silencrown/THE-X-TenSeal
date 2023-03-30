import numpy as np

class EncTransformer():
    """
    A original Transformer class of architecture
    """
    def __init__(self, torch_nn, torch_T):
        """
        Args: 
            torch_nn: torch model
            torch_T: softmax used assistant model T
        """
        self.torch_nn = torch_nn
        self.torch_T = torch_T
        pass

    def forward(self, enc_x):
        """
        Args:
            enc_x: input data
        """
        pass

    def forward(self, enc_x):
        pass

    @classmethod
    def self_attn(enc_x, K, V, Q):
        pass

    @classmethod
    def attn(enc_x, K, V, Q):
        pass

    @classmethod
    def softmax(enc_X, T):
        pass

    @classmethod
    def relu_enc(enc_X, tag_client):
        pass
