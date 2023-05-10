# THE-X NN

The-X NN is an extension to `torch.nn.Module` that adds several necessary FHE (fully homomorphic encryption) features to the class, allowing it to be used in constructing neural networks.

## Features

+ Adds `model.encrypt()` and `model.decrypt()` methods to the standard `model.train()` and `model.eval()`, allowing the user to choose between plain or encrypted data mode. When the data state is changed, the submodule and function in the model will also be changed into the specific state, which can be achieved by using the `TenSEAL` interface.
+ Supports built-in agent models for non-polynomial functions that cannot be directly changed into TenSEAL `CKKS` format. These agent models use approximation to convert the non-polynomial functions into polynomial functions.

## Usage

To use The-X NN, simply import the library and use it as you would any other PyTorch module. The added encrypt() and decrypt() methods can be used to switch between plain and encrypted data states, allowing for secure inference on encrypted data.

## Example

```python
from thex import xnn, cxt_man

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
x = np.arange(2 * 3 * 4).reshape([2, 3, 4])

# change the data into FHE
x_enc = cxt_man.encrypt(x)

# inference on the encrypted data
y_enc = model(x)

# get the plain result
y_dec = cxt_man.decrypt(y)
```

## Possible Implementation of xnn.Module

```python
import torch
import tenseal as ts


class Module(torch.nn.Module):
    """Wrapper class for torch.nn.Module to add FHE support."""

    def __init__(self, encryption_context=None):
        super(Module, self).__init__()
        self.is_encrypted = False
        self.encryption_context = encryption_context

    def encrypt(self, encryption_context=None):
        """Switches the module to FHE mode."""
        self.is_encrypted = True
        if not encryption_context:
            self.encryption_context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096, coeff_mod_bit_sizes=[40, 40, 40, 40])

    def decrypt(self):
        """Switches the module to plain mode."""
        self.is_encrypted = False
        self.encryption_context = None

    def forward(self, *inputs):
        """Abstract method that should be overridden by all subclasses."""
        raise NotImplementedError
```

## Opeartors

1. `xnn.Linear`: This is a fully-connected layer operator, which is used for the linear transformation of input data.
2. `xnn.MultiHeadAttention`: This operator computes self-attention for input data, which is an important part of the transformer model.
3. `xnn.LayerNorm`: This operator implements layer normalization, which is used in various parts of the transformer model.
4. `xnn.ReLU`: This is a rectified linear unit operator, which is used as the activation function in various parts of the model.
5. `xnn.Dropout`: This operator is used for regularization by randomly dropping out some of the input during training.
6. `xnn.TransformerEncoderLayer`: This operator defines a single layer of the transformer encoder, which can be used to build the full transformer encoder.
7. `xnn.TransformerDecoderLayer`: This operator defines a single layer of the transformer decoder, which can be used to build the full transformer decoder.
8. `xnn.Softmax`: This operator is used to compute the attention scores using the dot product of the queries and keys.
9. `xnn.Transpose`: This operator is used to transpose the output of a previous operator to match the expected shape.  [link](https://github.com/OpenMined/TenSEAL/pull/443)
10. `xnn.MaskedFill`: This operator is used to mask out the padded values in the input sequence.
11. `xnn.MatMul`: This operator is used for matrix multiplication. It's used to compute the final output of the attention mechanism.

## Workflow

1. pretrained model -> origin model
2. origin model -> converted model
3. converted model -> tenseal supported model

## TODO List

- [X] Operators
- [ ] bert-toy enc test
- [ ] bert-tiny convert
- [ ] bert-tiny enc test
- [ ] benchmark of each oprators
