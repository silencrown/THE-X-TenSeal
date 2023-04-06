class EncTransformer():
    """
    A original Transformer class of architecture
    """
    def __init__(self, torch_nn, torch_T):
        """
        Args: 
            torch_nn: torch model
            torch_T: softmax used assistant model T
        """
        self.torch_nn = torch_nn
        self.torch_T = torch_T
        pass

    def forward(self, enc_x):
        """
        Args:
            enc_x: input data
        """
        pass

    def forward(self, enc_x):
        pass

    @classmethod
    def self_attn(enc_x, K, V, Q):
        pass

    @classmethod
    def attn(enc_x, K, V, Q):
        pass


    class enc_softmax():
        def __init__(self, softmax_approx: SoftmaxApprox, relu=enc_relu):
            self.softmax_approx = softmax_approx
            # load weights
            self.fc1_weight = softmax_approx.fc1.weight.T.data.tolist()
            self.fc1_bias = softmax_approx.fc1.bias.data.tolist()
            self.fc2_weight = softmax_approx.fc2.weight.T.data.tolist()
            self.fc2_bias = softmax_approx.fc2.bias.data.tolist()
            self.fc2_weight = softmax_approx.fc2.weight.T.data.tolist()
            self.fc2_bias = softmax_approx.fc2.bias.data.tolist()

        def forward(self, enc_x):
            enc_x = enc_x / 2 + 1
            exp_score = self.enc_relu(enc_x * enc_x * enc_x)
            
            # fc1 layer
            enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
            enc_x = self.relu(enc_x)
            # fc2 layer
            enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias

            return enc_x
        
            t = input_tensor / 2 + 1
            exp_of_score = F.relu(t * t * t)
            x = exp_of_score.sum(-1, keepdim=True).unsqueeze(-1)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x).squeeze(dim=-1)
            return exp_of_score * x

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)


    @classmethod
    def relu_enc(enc_X, tag_client):
        pass
