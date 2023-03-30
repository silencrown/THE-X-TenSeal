import unittest
import torch

from src.utils import LoggingUtils
import src.operators as op

log = LoggingUtils(logger_name='softmax_logger')
log.add_console_handler()

class TestSoftmax(unittest.TestCase):
    def test_appr_softmax():
        """
        test softmax approximation.
        """
        # forward test
        x = torch.randn(10)
        log.debug(f"x: {x}")
        softmax_appr = op.SoftmaxApprox()
        log.debug(f"softmax_appr: {softmax_appr(x)}")

        # train test
        softmax_appr_trainer = op.SoftmaxApproxTrainer(softmax_appr, num_samples=1000000, input_size=128)
        # log.debug(f"generate train data: {softmax_appr_trainer._generate_train_data()[0][0]}")
        softmax_appr_trainer.train()

    def test_softmax():
        """
        Test softmax function.
        """
        x_torch = torch.rand(10, 10)
        x_np = x_torch.numpy()

        log.debug(f"torch: {x_torch[0][0]}")
        log.debug(f"numpy: {x_np[0][0]}")
    
        log.debug(f"softmax: {op.softmax(x_np)}")
        log.debug(f"softmax_torch: {op.softmax_torch(x_torch)}")

if __name__ == '__main__':
    unittest.main()