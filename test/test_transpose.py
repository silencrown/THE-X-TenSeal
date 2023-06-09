import tenseal as ts
import torch
import numpy as np
import unittest

import test_helper
from thex import logger, cxt_man, utils
from thex.xnn.transpose import transpose

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
    
    def test_multihead_transpose(self):
        # generate data
        shape = [2, 3, 4, 5]
        expected_shape = [2, 4, 3, 5]
        perm = [0, 2, 1, 3]
        torch_perm = [1, 2]

        data = np.arange(2*3*4*5).reshape(shape)
        logger(data)

        # numpy test
        np_trans = np.transpose(data, perm)
        # torch test
        pt_trans = np.array(torch.tensor(data).transpose(torch_perm[0], torch_perm[1]).tolist())

        # enc test
        enc_tensor = cxt_man.encrypt(data)
        assert enc_tensor.shape == shape

        enc_trans = enc_tensor.transpose(perm)
        assert enc_trans.shape == list(np_trans.shape)
        
        dec_trans = np.array(enc_trans.decrypt().tolist())

        # xnn test
        xnn_tensor = cxt_man.encrypt(data)
        xnn_trans = transpose(xnn_tensor, torch_perm)
        assert xnn_trans.shape == list(np_trans.shape)
        xnn_trans = np.array(xnn_trans.decrypt().tolist())

        logger(f"dec_result: {dec_trans}")
        logger(f"expected: {np_trans}")
        logger(f"shape {dec_trans.shape} {np_trans.shape} {pt_trans.shape} {xnn_trans.shape}")

        assert list(dec_trans.shape) == expected_shape
        assert np.allclose(pt_trans, np_trans, rtol=0, atol=0.01)
        assert np.allclose(dec_trans, np_trans, rtol=0, atol=0.01)
        assert np.allclose(xnn_trans, np_trans, rtol=0, atol=0.01)

    def _test_torch_example(self):
        pt_tensor = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        np_tensor = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        logger(pt_tensor)
        logger("------ torch transpose(-2, -1) ------")
        logger(pt_tensor.transpose(-2, -1))
        logger(np_tensor.transpose([0, 1, 3, 2]))
        logger("------ torch transpose(1, 2) -------")
        logger(pt_tensor.transpose(1, 2))
        logger(np_tensor.transpose([0, 2, 1, 3]))
        logger("------ torch transpose(0, 3) -------")
        logger(pt_tensor.transpose(0, 3))
        logger(np_tensor.transpose([3, 1, 2, 0]))
        logger("------ torch transpose(0, -1) -------")
        logger(pt_tensor.transpose(0, -1))
        logger(np_tensor.transpose([3, 1, 2, 0]))

    def _test_perm_convert(self):

        def is_right(input_tensor, torch_axes, np_axes):
            shape = list(input_tensor.shape)

            data = input_tensor
            # logger(data)

            # numpy test
            np_trans = np.transpose(data, np_axes)
            # torch test
            pt_trans = np.array(torch.tensor(data).transpose(torch_axes[0], torch_axes[1]).tolist())

            # enc test
            enc_tensor = cxt_man.encrypt(data)
            assert enc_tensor.shape == shape

            enc_trans = enc_tensor.transpose(list(np_axes))
            assert enc_trans.shape == list(np_trans.shape)

            dec_trans = np.array(enc_trans.decrypt().tolist())

            logger(f"dec_result: {dec_trans}")
            logger(f"expected: {np_trans}")
            logger(f"shape {dec_trans.shape} {np_trans.shape} {pt_trans.shape}")
            assert np.allclose(pt_trans, np_trans, rtol=0, atol=0.01)
            assert np.allclose(dec_trans, np_trans, rtol=0, atol=0.01) 
            
        input_array = np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
        transpose_axes = [2, 1]
        # output_array, target_axes = transpose_array(input_array, transpose_axes)
        target_axes = utils.get_axes_perm(input_array.shape, transpose_axes)
        logger(target_axes)
        is_right(input_array, transpose_axes, target_axes)


if __name__ == "__main__":
    unittest.main()
