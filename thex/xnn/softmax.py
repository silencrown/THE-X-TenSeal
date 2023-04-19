import torch
import torch.nn as nn
import torch.nn.functional as F

from thex import (
    logger,
    configer,
)
from .Module import FHELayer
from .linear import EncLinear

class ReciprocalApproximation(nn.Module):
    def __init__(self, hidden_size=16):
        super(ReciprocalApproximation, self).__init__()
        self.layer1 = nn.Linear(1, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class SoftmaxApprox(nn.Module):
    """
    Softmax approximation using linear network.

    Args:
    - hidden_size: hidden size of reciprocal approximation network.
    - use_pretrained: whether to use pretrained reciprocal model.
    - file_path: file path of pretrained reciprocal model.

    Raises:
    - ValueError: if pretrained model is not an instance of xnn.SoftmaxApprox.ReciprocalApproximation.
    """
    
    def __init__(self, hidden_size=16, use_pretrained=True, file_path=None):
        super(SoftmaxApprox, self).__init__()
        # init reciprocal approximation model
        self.reciprocal = ReciprocalApproximation(hidden_size=hidden_size)
        # load pretrained approximation model
        if use_pretrained:
            if file_path is None:
                file_path = configer()['softmax_approx']
                self.safe_load(file_path, self.reciprocal)
                logger.info(f"Load Softmax-Approximation Pretrained Model from: {file_path}")
    
    @staticmethod
    def safe_load(file_path, model):
        """
        Load model from file_path with parameter check.
        """
        state_dict = torch.load(file_path)

        for name, param in state_dict.items():
            if name in model.state_dict():
                if param.size() != model.state_dict()[name].size():
                    raise ValueError(f"Weight size mismatch: expected {model.state_dict()[name].size()} for {name}, but got {param.size()}")

        model.load_state_dict(state_dict)
        
    def forward(self, input_tensor):
        """
        implement of paper's approximation function: 
            $$
            S(x_i) = x_i * T ( \sum_j ReLU (((x_j)/2 + 1)^3))
            $$
        """
        x = input_tensor / 2 + 1
        exp_score = F.relu(x * x * x)
        exp_sum = exp_score.sum(-1, keepdim=True).unsqueeze(-1) # x -> [N, L, 1, H, 1]
        reci_exp_sum = self.reciprocal(exp_sum).squeeze(dim=-1) # [N, L, 1, H]

        return exp_score * reci_exp_sum

class EncReciprocalApproximation(FHELayer):
    def __init__(self, torch_nn=None):
        super(EncReciprocalApproximation, self).__init__()

        if torch_nn is None:
            torch_nn = ReciprocalApproximation()
            torch_nn.load_state_dict(torch.load(configer()['softmax_approx']))
            logger.info(f"Load Softmax Model from: {configer()['softmax_approx']}")

        self.layer1 = EncLinear(torch_nn.layer1)
        self.layer2 = EncLinear(torch_nn.layer2)
        self.layer3 = EncLinear(torch_nn.layer3)

    def forward(self, enc_x):
        enc_x = self.layer1(enc_x)
        enc_x = self.layer2(enc_x)
        enc_x = self.layer3(enc_x)
        return enc_x

class EncSoftmax(FHELayer):
    def __init__(self, torch_nn):
        super(EncSoftmax, self).__init__()
        self.enc_reciprocal = EncReciprocalApproximation(torch_nn.reciprocal)
        pass
        
    def forward(self, enc_x):
        # enc_x = enc_x.mm(self.fc_weight) + self.fc_bias
        pass

