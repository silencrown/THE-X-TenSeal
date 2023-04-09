import tenseal as ts
import torch
import torch.nn as nn
import torch.nn.functional as F

from thex.xnn.Module import FHELayer


class enc_Softmax(FHELayer):
    def __init__(self, torch_nn):
        # TODO: implement this
        super(enc_Softmax, self).__init__()
        
    def forward(self, enc_x):
        # enc_x = enc_x.mm(self.fc_weight) + self.fc_bias
        pass
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
        e = self.relu((input_tensor / 2 + 1) ** 3)
        x = e.sum(dim=-1, keepdim=True).unsqueeze(-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x).squeeze(dim=-1)
        return e * x
    
    def origin_forward(self, input_tensor):
        t = input_tensor / 2 + 1
        exp_of_score = F.relu(t * t * t)
        x = exp_of_score.sum(-1, keepdim=True).unsqueeze(-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x).squeeze(dim=-1)
        return exp_of_score * x