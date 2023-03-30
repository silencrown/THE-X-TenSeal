import tenseal as ts
import src.operators as op
import unittest

def _get_context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()
    return context