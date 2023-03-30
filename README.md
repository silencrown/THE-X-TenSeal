# THE-X NN
The-X NN is an extension to `torch.nn.Module` that adds several necessary FHE (fully homomorphic encryption) features to the class, allowing it to be used in constructing neural networks.

## Features
+ Adds `model.encrypt()` and `model.decrypt()` methods to the standard `model.train()` and `model.eval()`, allowing the user to choose between plain or encrypted data mode. When the data state is changed, the submodule and function in the model will also be changed into the specific state, which can be achieved by using the `TenSEAL` interface.
+ Supports built-in agent models for non-polynomial functions that cannot be directly changed into TenSEAL `CKKS` format. These agent models use approximation to convert the non-polynomial functions into polynomial functions.


## Usage
To use The-X NN, simply import the library and use it as you would any other PyTorch module. The added encrypt() and decrypt() methods can be used to switch between plain and encrypted data states, allowing for secure inference on encrypted data.

## Example
```
impoert tenseal as ts
from the_x import xnn

class Model(xnn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = xnn.MultiLayerPerceptron(10, 5, activation=xnn.ReLU())
        self.fc2 = xnn.MultiLayerPerceptron(5, 2, activation=xnn.ReLU())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Model()
model.encrypt()
x = xnn.randn(1, 10).encypt()
y = model(x)
y_decrypt = model.decrypt(y)
```



## Workflow
1. pretrained model -> origin model
2. origin model -> converted model
3. converted model -> tenseal supported model