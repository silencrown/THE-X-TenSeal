import unittest
import tenseal as ts

import test_helper
from thex import cxt_man
from thex import logger
from thex.ContextManager import ContextManager


class TestContextManager(unittest.TestCase):
    def setUp(self):
        self.context_manager = ContextManager(poly_mod=32768, inner_primes=35, precision_integer=5)

    def test_setup_context(self):
        ctx = self.context_manager.context()
        self.assertIsInstance(ctx, ts.Context)

    def test_depth_limiter(self):
        @self.context_manager.depth_limiter(depth_increment=1)
        def sqart(x):
            return x ** 2

        # Test with depth within limit
        result = sqart(2)
        self.assertEqual(self.context_manager.depth, 1)
    
    def _test_depth_refresher(self):
        @self.context_manager.depth_refresher()
        def update(x):
            pass
        self.assertEqual(self.context_manager.depth, 0)
    
    # TODO: test encrypt


if __name__ == '__main__':
    unittest.main()
