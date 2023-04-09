import sys
import os

the_x_parent_directory = os.path.abspath("/home/gaosq/the-X-TenSeal/")
if the_x_parent_directory not in sys.path:
    sys.path.insert(0, the_x_parent_directory)

import unittest
import tenseal as ts

from thex.ContextManager import ContextManager


class TestContextManager(unittest.TestCase):
    def setUp(self):
        self.context_manager = ContextManager(poly_mod=32768, inner_primes=35, precision_integer=5)

    def test_get_context(self):
        ctx, precision = self.context_manager.get_context()
        self.assertIsInstance(ctx, ts.Context)
        self.assertEqual(precision, 8192)

    def test_depth_limiter(self):
        @self.context_manager.depth_limiter(depth_increment=1)
        def sqart(x):
            return x ** 2

        # Test with depth within limit
        result = sqart(2)
        self.assertEqual(self.context_manager.depth, 1)
    
    def test_depth_updater(self):
        @self.context_manager.depth_updater()
        def update(x):
            pass
        self.assertEqual(self.context_manager.depth, 0)
    
    # TODO: test encrypt


if __name__ == '__main__':
    unittest.main()
