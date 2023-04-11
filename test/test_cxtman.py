import unittest

import numpy as np
import tenseal as ts

import test_helper
from thex import cxt_man
from thex import logger


class TestContextManager(unittest.TestCase):
    def _test_setup_context(self):
        cxt = cxt_man.context
        self.assertIsInstance(cxt, ts.Context)

    def _test_encrypt(self):
        # encrypt a vector
        vector = np.arange(10)
        enc_vector = cxt_man.encrypt(vector)
        self.assertIsInstance(enc_vector, (ts.CKKSVector, ts.CKKSTensor))

        # decrypt the vector
        dec_vector = cxt_man.decrypt(enc_vector)
        logger(f"vector: {vector}")
        logger(f"dec_vector: {dec_vector}")
        np.testing.assert_array_almost_equal(vector, dec_vector, decimal=2)
        
        # encrypt a tensor
        tensor = np.arange(1, 128, 2).reshape(2, -1)
        enc_tensor = cxt_man.encrypt(tensor)
        self.assertIsInstance(enc_tensor, ts.CKKSTensor)

        # decrypt the tensor
        dec_tensor = cxt_man.decrypt(enc_tensor)
        logger(f"tensor: {tensor}")
        logger(f"dec_tensor: {dec_tensor}")
        np.testing.assert_array_almost_equal(tensor, dec_tensor, decimal=2)

    @cxt_man.depth_limiter(depth_increment=1)
    def he_sqart(self, enc_x):
        return enc_x ** 2
    
    def sqart(self, x):
        return x ** 2

    @cxt_man.depth_refresher()
    def _he_vector_refresh(self, enc_x):
        x = cxt_man.decrypt(enc_x)
        enc_x = cxt_man.encrypt(x)
        logger("context manager refreshed...")
        return enc_x

    def _multi_vector(self):
        # create a encrypted tensor
        vector = np.arange(10)
        enc_vector = cxt_man.encrypt(vector)

        for _ in range(cxt_man.max_depth-1):
            enc_vector = self.he_sqart(enc_vector)
            logger(f"enc_vector: {enc_vector.decrypt()}")
        
        for _ in range(cxt_man.max_depth-1):
            vector = self.sqart(vector)
            logger(f"vector: {vector}")
        
        return vector, enc_vector

    def _test_depth_limiter(self):
        """Test depth limiter"""
        _, enc_vector = self._multi_vector()
        # call the multi function and expect a `max depth exceeded` ValueError
        with self.assertRaises(ValueError):
            err_vector = self.he_sqart(enc_vector)

    def test_depth_refresher(self):
        """Test depth refresher"""
        _, enc_vector = self._multi_vector()
        logger(f"current depth: {cxt_man.depth}")
        self._he_vector_refresh(enc_vector)
        logger(f"current depth: {cxt_man.depth}")
        self.assertEqual(cxt_man.depth, 0)

if __name__ == '__main__':
    unittest.main()