import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58],
                   [102, 43, 37], [69, 96, 70], [73, 67, 43],
                   [91, 88, 64], [87, 134, 58], [102, 43, 37],
                   [69, 96, 70], [73, 67, 43], [91, 88, 64],
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]],
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133],
                    [22, 37], [103, 119], [56, 70],
                    [81, 101], [119, 133], [22, 37],
                    [103, 119], [56, 70], [81, 101],
                    [119, 133], [22, 37], [103, 119]],
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

train_ds = TensorDataset(inputs,targets)
# print(train_ds[0:3])

# split the dataset into batch
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# use for-in to visit batch
for xb, yb in train_dl:
    print(xb)
    print(yb)

# define model through nn.Liner
inputs_size = 3
targets_size = 2

# generate weight & bias automatically
model = nn.Linear(inputs_size, targets_size)
# print(model.weight)
# print(model.bias)
# print(list(model.parameters()))

# generate predictions
preds = model(inputs)

# loss function
loss = F.mse_loss(preds, targets)
loss_fn = F.mse_loss
print(loss)

# define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


# train
def train(epochs_num, model, loss_fn, opt):
    for epoch in range(epochs_num):
        for xb, yb in train_dl:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        if (epoch+1) % 10 == 0:
            print('Epoch: ', epoch+1, 'Loss: ', loss.data.numpy())


if __name__ == '__main__':
    train(1000, model, loss_fn, opt)
    print(model(inputs))
    torch.save(model, 'checkpoint/my.pt')
