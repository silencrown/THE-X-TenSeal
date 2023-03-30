# encoding=utf8
"""
environment test
"""

import unittest

class EnviromentTest(unittest.TestCase): 
    def test_cuda(self):
        import torch
        result = torch.cuda.is_available()
        self.assertEqual(result, True)

    def test_tenseal(self):
        # import tenseal
        pass

if __name__ == '__main__':
    unittest.main()