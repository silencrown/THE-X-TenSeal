import tenseal as ts
import torch
import numpy as np
import unittest

import test_helper
from thex import logger, cxt_man


class TestTranspose(unittest.TestCase):
    
    def _test_2D_transpose(self):
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
    
    def _test_3D_transpose(self):
        # generate data
        # data = np.array([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]], [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]]])
        data = np.arange(24)
        data = data.reshape(2, 3, 4)
        logger(data)
        shape = [2, 3, 4]
        # expected data
        expected = np.transpose(np.array(data).reshape(shape))
        # FIXME: torch.transpose(-2, -1) is not like ts.CKKSTensor.transpose() 
        pt_tensor = torch.tensor(data)
        pt_tensor = pt_tensor.transpose(-2, -1)
        pt_expected = np.array(pt_tensor.tolist())
        assert np.allclose(pt_expected, expected, rtol=0, atol=0.01)
        
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

    def test_torch_example(self):
        pt_tensor = torch.arange(24).reshape(2, 3, 4)
        logger(pt_tensor)
        pt_tensor = pt_tensor.transpose(-2, -1)
        logger("------ torch transpose(-2, -1) ------")
        logger(pt_tensor)


if __name__ == "__main__":
    unittest.main()
