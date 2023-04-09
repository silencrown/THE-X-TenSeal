from __future__ import annotations

import math
import random
import jsonpickle
import time

import numpy as np
import tenseal as ts


def test_tenseal_ckks() -> None:

    # 8192 under 128 security level -> 218
    # 16384 under 128 security level -> 438
    # 32768 under 128 security level -> 881
    # from MS SEAL Manuscript
    _table = {
        1024: 27,
        2048: 54,
        4096: 109,
        8192: 218,
        16384: 438,
        32768: 881,
    }
    _poly_mod = 32768
    # need >= 30? larger size, more accurate, less size of multiplication depth
    _size_inner_primes = 35
    # bit precision before decimal point
    _precision_integer = 5
    # bit precision after decimal point
    _precision_fractional = _size_inner_primes - _precision_integer
    _scale = 2**_size_inner_primes
    # depth
    _depth = math.floor(
        (_table[_poly_mod] - (_precision_integer + _size_inner_primes) * 2)
        / _size_inner_primes
    )
    print("multiplication (theoretical) depth: {}".format(_depth))
    # restriction: sum(_coeff_mod_bit_sizes) < _table[_poly_mod]
    _coeff_mod_bit_sizes = (
        [_precision_integer + _size_inner_primes]
        + [_size_inner_primes for _ in range(_depth)]
        + [_precision_integer + _size_inner_primes]
    )
    # SEAL context
    ctx = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=_poly_mod,
        coeff_mod_bit_sizes=_coeff_mod_bit_sizes,
    )
    ctx.global_scale = _scale

    # max input size
    # theoretically, the max input size could be `_poly_mod` / 2
    # here, if / 2, the last 4-5 depths of multiplication will meet noise explosion, it may caused by lower precision setting.
    _max_size_vec = int(_poly_mod / 4)
    _size_vec = 128
    print(f"max input size: {_max_size_vec}")
    print(f"input size: {_size_vec}")

    # generate input vector
    # _vec = [random.random() for _ in range(_size_vec)]
    # _vec_ct = ts.ckks_vector(ctx, _vec)
    # _vec_np = np.array(_vec)
    # print("original vec: {} \n\n".format(_vec))
    # print("ciphertext size: {} mb".format(
    #         len(jsonpickle.encode(_vec_ct.serialize())) / 1e6
    #     )
    # )

    # multiplication test
    # for i in range(_depth):
    #     print("="*20 + f" {i}-th multiplication " + "="*20)
    #     # test time
    #     start_time = time.time()
    #     _vec_ct = _vec_ct * _vec_ct
    #     print(f"time cost: {time.time()-start_time:.3f}s")
    #     # test precision
    #     _vec_np = _vec_np * _vec_np
    #     _vec_ct_dec = _vec_ct.decrypt()
    #     _diff = sum(
    #         [abs(_vec_ct_dec[i] - _vec_np[i]) for i in range(len(_vec_ct_dec))]
    #     ) / len(_vec_ct_dec)
    #     print("avg difference (precision) ={}\n\n".format(_diff))

    # test tensor (seams slower than vector setting?)
    _size_shape = 8
    _a1 = np.random.rand(_size_shape, _size_shape)
    _a2 = np.random.rand(_size_shape, _size_shape)
    _tensor1 = ts.plain_tensor(_a1)
    _tensor2 = ts.plain_tensor(_a2)
    _tensor1_ct = ts.ckks_tensor(ctx, _tensor1)
    _tensor2_ct = ts.ckks_tensor(ctx, _tensor2)
    for i in range(_depth):
        _a1 *= _a2
        _tensor1_ct *= _tensor2_ct
        _tensor1_dec = np.array(_tensor1_ct.decrypt().tolist()).reshape(
            _tensor1_ct.shape
        )
        _diff = _a1 - _tensor1_dec
        _diff_avg = np.sum(np.absolute(_diff)) / _diff.size
        print("{}-th multiplication depth".format(i))
        # print("{}-th mul ckks res:{}".format(i, _tensor1_dec))
        # print("{}-th mul expt res:{}".format(i, _a1))
        print("avg difference (precision) ={}\n\n".format(_diff_avg))


if __name__ == "__main__":
    test_tenseal_ckks()
