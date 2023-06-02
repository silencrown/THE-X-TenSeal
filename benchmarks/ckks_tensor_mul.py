import tenseal as ts
import numpy as np
import pytest
from memory_profiler import memory_usage

# from ckks_tenseal import _almost_equal as almost_equal


SHAPE = [
    [2, 128], 
    [4, 128], 
    [8, 128], 
    [16, 128], 
    [32, 128], 
    [64, 128], 
    [128, 128]
]
POLY_COEFF = [
    # (2048, [54]), 
    # (2048, [31, 23]), 
    (4096, [30, 22, 22, 30]), 
    (8192, [30, 24, 24, 24, 24, 24, 24, 24, 24, 30]), 
    (8192, [30, 24, 24, 24, 24, 24, 30]), 
    (8192, [30, 24, 24, 24, 24, 24, 24, 30]),
    (8192, [30, 24, 24, 24, 24, 24, 24, 24,30]), 
]

# encryption
def _enc(shape, poly_mod, coeff_mod, is_trans):
    
    data = np.around(np.random.rand(*shape), 4)
    if is_trans:
        data = data.T
    data = data.tolist()

    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod, coeff_mod)
    context.generate_galois_keys()
    context.global_scale = 2**30
    # context.auto_mod_switch = False
    ct = ts.ckks_tensor(context, data)
    return ct

@pytest.mark.parametrize(
    "shape", 
    SHAPE
)
@pytest.mark.parametrize(
    "poly_mod, coeff_mod", 
    POLY_COEFF
)
def test_enc(benchmark, shape, poly_mod, coeff_mod):

    # encryption
    def _enc(shape, poly_mod, coeff_mod, is_trans):
        
        data = np.random.rand(*shape)
        if is_trans:
            data = data.T
        data = data.tolist()

        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod, coeff_mod)
        context.global_scale = 2**40
        # context.auto_mod_switch = False
        ct = ts.ckks_tensor(context, data)
        return ct
    
    def _enc_wrapper():
        _enc(shape, poly_mod, coeff_mod, False)

    # get result
    mem_usage = memory_usage(_enc_wrapper)
    print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
    print('Maximum memory usage: %s' % max(mem_usage))

    benchmark(_enc_wrapper)

@pytest.mark.parametrize(
    "shape", 
    SHAPE
)
@pytest.mark.parametrize(
    "poly_mod, coeff_mod", 
    POLY_COEFF
)
def test_ct_mm(benchmark, shape, poly_mod, coeff_mod):
    ct1 = _enc(shape, poly_mod, coeff_mod, False)
    ct2 = _enc(shape, poly_mod, coeff_mod, True)

    # encryption
    def _mm(ct1, ct2):
        ct1.mm(ct2)
        return ct1
    
    def _mm_wrapper():
        _mm(ct1, ct2)

    # get result
    mem_usage = memory_usage(_mm_wrapper)
    print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
    print('Maximum memory usage: %s' % max(mem_usage))

    benchmark(_mm_wrapper)