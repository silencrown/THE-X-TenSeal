import tenseal as ts
import numpy as np
import unittest

import test_helper
from thex import logger, cxt_man


class TestTranspose(unittest.TestCase):
    
    def test_cxt_transpose(self):
        # generate data
        data = np.array([[1, 2, 3], [4, 5, 6]])
        shape = [2, 3]
        # expected data
        expected = np.transpose(np.array(data).reshape(shape))
        # test enc result
        enc_tensor = cxt_man.encrypt(data)
        assert enc_tensor.shape == shape
        # test enc transpose 
        enc_result = enc_tensor.transpose()
        assert enc_result.shape == list(expected.shape)
        # test dec result
        dec_result = np.array(enc_result.decrypt().tolist())
        assert np.allclose(dec_result, expected, rtol=0, atol=0.01)
        logger(f"dec_result: {dec_result}")
        logger(f"expected: {expected}")


if __name__ == "__main__":
    unittest.main()
