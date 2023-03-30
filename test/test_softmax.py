import torch
import torch.nn as nn
import tenseal as ts
import numpy as np
import time

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(0, keepdim=True)
    return X_exp / partition

def softmax_appr(X, net):
    pass

def softmax_enc(X, T):
    pass

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def main():
    # generate evaluate samples
    hidden_size = 512
    X = np.random.normal(size=(1000, 512))
    X = np.clip(X, -3, 3)
    # print(X)
    Y = softmax(torch.tensor(X))
    # print(Y)
    batch_size = 10
    data_iter = load_array((X, Y), batch_size)

    # init net
    net = nn.Sequential(nn.Linear(512, 1024), 
                        nn.Linear(1024, 1024), 
                        nn.Linear(1024, 512))
    net.apply(init_weights)




if __name__ == "__main__":
    main()
