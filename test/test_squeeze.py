import tenseal as ts
import thex.xnn as xnn
import unittest

def _setup_context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()
    return context

class TestSqueeze(unittest.TestCase):
    def test_squeeze(self):
        context = _setup_context()
        ckks_encoder = ts.CKKSEncoder(context)
        scale = 2**40
        vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        vector = ckks_encoder.encode(vector, scale)
        vector = xnn.squeeze(context, vector, 2, 2)
        vector = ckks_encoder.decode(vector)
        self.assertEqual(vector, [3.0, 4.0, 5.0, 6.0])
