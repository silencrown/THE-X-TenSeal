import unittest
import torch
import numpy as np

import test_helper
from thex import cxt_man
from thex import logger

from thex.xnn.softmax import (
    SoftmaxApprox,
    EncSoftmax
)
from thex.convert.softmax_approx import (
    SoftmaxApproxTrainer,
)

class TestSoftmax(unittest.TestCase):
    def softmax(self, x):
        return torch.nn.Softmax(dim=-1)(x)
    
    def _train_softmax_approx(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        while True:
            # generate model
            model = SoftmaxApprox(use_pretrained=False, 
                                  hidden_size=32)
            model.to(device)
            # generate trainer
            torch.seed()
            trainer = SoftmaxApproxTrainer(model, 
                                           batch_size=128,
                                           num_epochs=100000,
                                           lr=2e-6,
                                           input_size=128, 
                                           device=device)
            logger("=========================================")
            logger(f"trainer epochs: {trainer.num_epochs}")
            logger(f"softmax reci size: {model.reciprocal.hidden_size}")
            logger("=========================================")
            if trainer.train() < 1e-3: break
        trainer.save()
    
    def _test_softmax_approx(self):
        # self.train_softmax_approx()
        # generate input tensor
        input_tensor = torch.randn(1, 2, 128, 128) * 3.0
        # generate label
        label = self.softmax(input_tensor)
        # generate model (default: use pretrained model)
        model = SoftmaxApprox(use_pretrained=True,
                              hidden_size=64, 
                              file_path="../cache/softmax_approx_8e-4_64.pt")
        # get result
        result = model(input_tensor)
        # compare result with label
        logger(f"result: {result[0, 0, 0, 0]} {result[0, 0, 0, 1]} {result[0, 0, 0, 2]}")
        logger(f"label: {label[0, 0, 0, 0]} {label[0, 0, 0, 1]} {label[0, 0, 0, 2]}")
        logger(f"MSE: {torch.nn.MSELoss()(result.view(-1), label.view(-1))}")
        self.assertTrue(torch.allclose(result, label, atol=1e-1))

    def test_enc_softmax(self):
        input_tensor = torch.randn(2, 128)

        enc_tensor = cxt_man.encrypt(input_tensor)
        model = EncSoftmax()

        # use numpy to compare
        label = self.softmax(input_tensor).numpy()
        result = np.array(cxt_man.decrypt(model(enc_tensor)))
        
        logger(f"mse: {np.mean(np.square(result - label))}")

if __name__ == '__main__':

    unittest.main()
