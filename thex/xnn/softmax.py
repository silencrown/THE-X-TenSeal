import torch
import torch.nn as nn
import torch.nn.functional as F

from thex.xnn.Module import FHELayer


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
    """
    def __init__(self, hidden_size=16):
        super(SoftmaxApprox, self).__init__()
        self.reciprocal = ReciprocalApproximation(hidden_size=hidden_size)
    
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
    def __init__(self, torch_nn):
        super(EncReciprocalApproximation, self).__init__()
        pass
    def forward(self, enc_x):
        pass

class EncSoftmax(FHELayer):
    def __init__(self):
        super(EncSoftmax, self).__init__()
        pass
        
    def forward(self, enc_x):
        # enc_x = enc_x.mm(self.fc_weight) + self.fc_bias
        pass

