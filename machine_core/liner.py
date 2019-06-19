# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Weught & Bias
w = torch.rand(2,3,requires_grad=True)
b = torch.rand(2,requires_grad=True)


# inputs * weight + bias
def model(x):
    # return x @ w.t() + b
    return torch.mm(x,w.t()) + b


# loss function
def mse(t1,t2):
    diff = t1-t2
    return torch.sum(diff * diff) / diff.numel()


pred = model(inputs)
print(pred)

loss = mse(pred,targets)
# backward propration
loss.backward()

print(loss)

# train
for i in range(10000):
    pred = model(inputs)
    loss = mse(pred,targets)
    loss.backward()
    # gradient descent
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

print(loss)
print(pred)

if __name__ == '__main__':
    model = torch.load('checkpoint/my.pt')
    pred = model(inputs)
    print(pred)
    print(F.mse_loss(pred, targets))
