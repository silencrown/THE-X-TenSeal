import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from thex import logger


def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def softmax_torch(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class SoftmaxApprox(nn.Module):
    """
    Softmax approximation using linear network.
    """
    def __init__(self, relu=F.relu, hidden_size=64):
        """
        Initialize softmax approximation.
        Args:
            relu: relu function, `encrypted_support_relu` or `origin_relu`
            input_size: input size, should be equal with transformer `attention_size`
            hidden_size: hidden size
            output_size: output size (default == input_size)
        """
        super(SoftmaxApprox, self).__init__()
        self.relu = relu
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)


    def forward(self, input_tensor):
        """
        S(xi) = xi * T (∑_j ReLU(((xj)/2 + 1)^3))
        """
        t = input_tensor / 2 + 1
        exp_of_score = F.relu(t * t * t)
        x = exp_of_score.sum(-1, keepdim=True).unsqueeze(-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x).squeeze(dim=-1)
        return exp_of_score * x
        # e = self.relu((input_tensor / 2 + 1) ** 3)
        # x = e.sum(dim=-1, keepdim=True).unsqueeze(-1)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.fc3(x).squeeze(dim=-1)
        # return e * x
    
    def origin_forward(self, input_tensor):
        t = input_tensor / 2 + 1
        exp_of_score = F.relu(t * t * t)
        x = exp_of_score.sum(-1, keepdim=True).unsqueeze(-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x).squeeze(dim=-1)
        return exp_of_score * x
 

class SoftmaxApproxTrainer():
    def __init__(self, softmodel: SoftmaxApprox, num_samples=1e6, input_size=128, batch_size=1, lr=0.0001, num_epochs=100):
        self.softmodel = softmodel
        self.num_samples = num_samples
        self.input_size = input_size
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

    def _generate_train_data(self):
        """
        Generate training data for softmax approximation's linear network.
        """
        x = (torch.rand(self.num_samples, self.input_size) * 6) - 3
        x.requires_grad = False
        y = softmax_torch(x)
        y.requires_grad = False
        return x, y
    
    def train(self):
        x, y = self._generate_train_data()
        writer = SummaryWriter(log_dir='logs')
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.softmodel.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            # TODO: batch
            optimizer.zero_grad()
            l = loss(self.softmodel(x), y)
            l.sum().backward()
            optimizer.step()
            logger.info(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
            writer.add_scalar('loss', float(l.sum()), epoch)
        writer.close()

    def save(self, file_path="output/softmax_approx.model"):
        """
        Save model.
        """
        torch.save(self.softmodel.state_dict(), file_path)
        logger.info(f"Softmax Model Saved in: {file_path}")
        return file_path
