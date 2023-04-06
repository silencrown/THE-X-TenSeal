import numpy as np
import unittest
import torch
import tenseal as ts

import test_helper
from thex.xnn.layernorm import enc_layernorm


class TestLayerNorm(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = torch.nn.Linear(input_size, input_size)
    
    def forward(self, x):
        return self.fc(x)

def test_layernorm():

    # generate a torch model
    torch_model = TestLayerNorm(10)
    enc_model = enc_layernorm(torch_model)
    print(enc_model.fc_weight)

    """ Initial Encryption Parameters """
    # controls precision of the fractional part
    bits_scale = 26
    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )
    # set the scale
    context.global_scale = pow(2, bits_scale)
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    # generate a random input
    x = np.random.rand(1, 10).astype(np.float32)
    enc_x = ts.ckks_tensor(context, x.tolist())

    # test the encrypted model
    enc_y = enc_model(enc_x)
    print(np.array(enc_y.decrypt().tolist()))


if "__main__" == __name__:
    test_layernorm()
