import tenseal as ts
import torch

import test_helper
from thex import ContextManager

def test_manager():
    # Setup TenSEAL context
    context_manager = ContextManager()
    context = context_manager.get_context()[0]

    a = torch.randn(64, 128)
    b = torch.randn(128, 64)

    # encrypted vectors
    print("start encrypt")
    enc_a = ts.ckks_tensor(context, a)
    enc_b = ts.ckks_tensor(context, b)

    print("matmul")
    result = enc_a.mm(enc_b)
    result.decrypt()

    print("test result")
    print(result)
    print(a.mm(b))


def main():
    context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
    context.generate_galois_keys()
    context.global_scale = 2**40
    

    a = torch.randn(64, 128)
    b = torch.randn(128, 64)

    # encrypted vectors
    print("start encrypt")
    enc_a = ts.ckks_tensor(context, a)
    enc_b = ts.ckks_tensor(context, b)

    print("matmul")
    result = enc_a.mm(enc_b)
    result.decrypt()

    print("test result")
    print(result)
    print(a.mm(b))

if __name__ == "__main__":
    test_manager()