import torch
from torch import nn
import tenseal as ts
import numpy as np
import random
import time
import sys

from src.utils import get_tenseal_context


class Server(object):
    def __init__(self):
        pass

    def relu(ckks_vector, client):
        return Client.relu(ckks_vector)
        
class Client(object):
    def __init__(self, ckks_context):
        self.ckks_context = ckks_context
    
    def relu(ckks_vector):
        ckks_vector()

def sqrt_test():     
    # hyper parameters 
    num_inputs = 128  
    num_outputs = 32  
    num_hiddens = 256 

    # init weight params
    W1 = torch.randn(num_inputs, num_hiddens, requires_grad=False) * 0.01
    b1 = torch.zeros(num_hiddens, requires_grad=False) 
    W2 = torch.randn(num_hiddens, num_outputs, requires_grad=False) * 0.01
    b2 = torch.zeros(num_outputs, requires_grad=True)
    params = [W1, b1, W2, b2]

    def relu(X):
        a = torch.zeros_like(X) 
        return torch.max(X, a) 
    
    def relu_enc(X):
        a = torch.zeros_like(X)


    # net model
    def net(X):
        X = X.reshape((-1, num_inputs)) 
        H = relu(X@W1 + b1) 
        return (H@W2 + b2) 

    enc_x = torch.randn(128)*0.01
    enc_x = enc_x.tolist()

    bits_scale = 26
    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
        # coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60]
    )

    # set the scale
    context.global_scale = pow(2, bits_scale)
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()
    enc_x = ts.ckks_vector(context, enc_x)
    start_time = time.time()
    enc_x = enc_x.mm(W1) + b1
    print(f"sqrt time cost: {time.time()-start_time:2f}s")

def relu(X):
    return np.max(X, np.zeros_like(X))

def relu_enc(X_enc, context):
    X = X_enc.decrypt()
    X_out = relu(X)
    return ts.X_out

def relu_test():
    context, max_size = get_tenseal_context()
    vec = [random.random() for _ in range(max_size)]
    vec_ct = ts.ckks_vector(context, vec)
    vec_np = np.array(vec)

    start_time = time.time()
    enc_x = enc_x.decrypt()
    print(enc_x[0])
    enc_x = ts.ckks_vector(context, enc_x)
    print(enc_x) 
    relu(vec_np)
    relu_enc(vec_ct)

def relu_test_old():
    enc_x = torch.randn(256)*0.01
    enc_x = enc_x.tolist()
    print(enc_x[0])
    print(sys.getsizeof(enc_x))
    bits_scale = 26
    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
        # coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60]
    )

    # set the scale
    context.global_scale = pow(2, bits_scale)
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()
    enc_x = ts.ckks_vector(context, enc_x)
    start_time = time.time()
    enc_x = enc_x.decrypt()
    print(enc_x[0])
    enc_x = ts.ckks_vector(context, enc_x)
    print(enc_x)
    
    print(f"relu time cost: {time.time()-start_time:2f}s")

def main():
    # relu_test()
    relu_test_old()
    # sqrt_test()


if __name__ == "__main__":
    main()