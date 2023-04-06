import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import tenseal as ts

from thex.models.cnn.cnn import (
    ConvNet, 
    train, 
    test
)
from thex.models.cnn.enc_cnn import (
    EncConvNet, 
    enc_test, 
)

def test_enc():
    """ Train Model """
    torch.manual_seed(73)
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    batch_size = 64

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = train(model, train_loader, criterion, optimizer, 10)
    test(model, test_loader, criterion)

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

    """ Enc Model Test """
    # Load one element at a time
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    # required for encoding
    kernel_shape = model.conv1.kernel_size
    stride = model.conv1.stride[0]
    enc_model = EncConvNet(model)
    enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)

if __name__ == "__main__":
    test_enc()