import unittest
import torch

from thex import operators as op
from thex import logger
class TestSoftmax(unittest.TestCase):
    def test_appr_softmax():
        """
        test softmax approximation.
        """
        # forward test
        x = torch.randn(10)
        logger(f"x: {x}")
        softmax_appr = op.SoftmaxApprox()
        logger(f"softmax_appr: {softmax_appr(x)}")

        # train test
        softmax_appr_trainer = op.SoftmaxApproxTrainer(softmax_appr, num_samples=1000000, input_size=128)
        # log(f"generate train data: {softmax_appr_trainer._generate_train_data()[0][0]}")
        softmax_appr_trainer.train()

    def test_softmax():
        """
        Test softmax function.
        """
        x_torch = torch.rand(10, 10)
        x_np = x_torch.numpy()

        logger(f"torch: {x_torch[0][0]}")
        logger(f"numpy: {x_np[0][0]}")
    
        logger(f"softmax: {op.softmax(x_np)}")
        logger(f"softmax_torch: {op.softmax_torch(x_torch)}")

if __name__ == '__main__':
    unittest.main()