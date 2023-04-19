import torch
import torch.nn as nn
import torch.nn.functional as F

from thex import logger
from thex.xnn.softmax import SoftmaxApprox


class SoftmaxApproxTrainer():
    def __init__(self, model, num_samples=1e6, input_size=128, batch_size=1, lr=0.0001, num_epochs=100):
        if not isinstance(model, SoftmaxApprox):
            raise ValueError("model must be an instance of xnn.SoftmaxApprox")
        self.model = model
        self.num_samples = num_samples
        self.input_size = input_size
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

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

        for epoch in range(self.num_epochs):
            input_tensor = torch.randn(self.batch_size, 2, self.input_size, self.input_size)
            label  = nn.Softmax(dim=-1)(input_tensor)
            pred = self.model(input_tensor)
            
            loss = nn.MSELoss()(pred.view(-1), label.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger(f'epoch {epoch + 1}, loss {float(loss.item()):.5f}')

    def save(self, file_path="output/softmax_approx.model"):
        """
        Save model.
        """
        torch.save(self.model.reciprocal.state_dict(), file_path)
        logger.info(f"Softmax Model Saved in: {file_path}")
        return file_path
