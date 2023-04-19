import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from thex import (
    logger,
    configer,
)
from thex.xnn.softmax import SoftmaxApprox


class SoftmaxApproxTrainer():
    def __init__(self, model, num_samples=1e6, input_size=128, batch_size=128, lr=1e-5, num_epochs=1000, device='cpu'):
        
        if not isinstance(model, SoftmaxApprox):
            raise ValueError("model must be an instance of xnn.SoftmaxApprox")
        
        self.model = model
        self.num_samples = num_samples
        self.input_size = input_size
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device

    def _generate_train_data(self):
        """
        Generate training data for softmax approximation's linear network.
        """
        x = (torch.randn(self.num_samples, self.input_size) * 6) - 3
        x.requires_grad = False
        y = nn.Softmax(x)
        y.requires_grad = False
        return x, y
    
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        iter_bar = tqdm.tqdm(range(self.num_epochs))
        for t in iter_bar:
            input_tensor = torch.randn(self.batch_size, 2, self.input_size, self.input_size, device=self.device) * 3.0
            label  = nn.Softmax(dim=-1)(input_tensor)
            pred = self.model(input_tensor)
            
            loss = nn.MSELoss()(pred.view(-1), label.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_bar.set_description("loss: %.4f" % loss.item())

        # return loss
        return loss

    def save(self, file_path=None):
        """
        Save model.
        """
        if file_path is None:
            file_path = configer()['softmax_approx']
        if self.device == 'cuda':
            self.model.cpu()
        torch.save(self.model.reciprocal.state_dict(), file_path)
        logger.info(f"Softmax Model Saved in: {file_path}")
        return file_path
