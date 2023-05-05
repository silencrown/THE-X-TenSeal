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
    
    def test_3D_transpose(self):
        # generate data
        shape = [2, 3, 4]
        expected_shape = [2, 4, 3]

        data = np.arange(24).reshape(shape)
        logger(data)

        # numpy test
        np_trans = np.transpose(data, [0, 2, 1])
        # torch test
        pt_trans = np.array(torch.tensor(data).transpose(-2, -1).tolist())

        # enc test
        enc_tensor = cxt_man.encrypt(data)
        assert enc_tensor.shape == shape

        enc_trans = enc_tensor.transpose([0, 2, 1])
        assert enc_trans.shape == list(np_trans.shape)
        
        dec_trans = np.array(enc_trans.decrypt().tolist())

        logger(f"dec_result: {dec_trans}")
        logger(f"expected: {np_trans}")
        logger(f"shape {dec_trans.shape} {np_trans.shape} {pt_trans.shape}")
        assert list(dec_trans.shape) == expected_shape
        assert np.allclose(pt_trans, np_trans, rtol=0, atol=0.01)
        assert np.allclose(dec_trans, np_trans, rtol=0, atol=0.01)

    def _test_torch_example(self):
        pt_tensor = torch.arange(24).reshape(2, 3, 4)
        logger(pt_tensor)
        pt_tensor = pt_tensor.transpose(-2, -1)
        logger("------ torch transpose(-2, -1) ------")
        logger(pt_tensor)


if __name__ == "__main__":
    unittest.main()
