import tenseal as ts
import torch
import time

import test_helper
from thex import cxt_man

def test_manager():
    a = torch.randn(64, 128)
    b = torch.randn(128, 64)

    # encrypted vectors
    print("start encrypt")
    enc_a = cxt_man.encrypt(a)
    enc_b = cxt_man.encrypt(b)

    start_time = time.time()
    result = enc_a.mm(enc_b)
    result.decrypt()

    print("test result")
    print(result)
    print(a.mm(b))
    print(f"encrypt time: {time.time() - start_time}")

def main():
    context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
    context.generate_galois_keys()
    context.global_scale = 2**40

if __name__ == "__main__":
    test_manager()