import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import sampler
from torchvision.datasets import MNIST

# # MNIST dataser (image & label)
# # # Download dataset 60000
# # dataset = MNIST(root='data/', download=True)
# # # print(len(dataset))
# #
# # # test data 10000
# # test_dataset = MNIST(root='data/', train=False)
# # # print(len(test_dataset))
# #
# # # contain a 28*28 image & a int label
# # print(dataset[0])
# #
# # image, label = dataset[0]
# # plt.imshow(image)
# # print('label:', label)
# # plt.waitforbuttonpress(1)

DOWNLOAD_MNIST = False

tensor_dataset = MNIST(root='data/',
                       train=True,
                       transform=transforms.ToTensor(),
                       download=DOWNLOAD_MNIST)

# image_tensor, label_tensor = tensor_dataset[0]
# print(image_tensor.shape, label_tensor)
# plt.imshow(image_tensor[0], cmap='gray')
# plt.waitforbuttonpress(10)


# Split validation set randomly
def split_indices(n, val_percent):
    # Calculate the size of validation set
    n_val = int(n*val_percent)
    # Create random permutation of 0 to n-1
    indices_random = np.random.permutation(n)
    # return train_dataset, val_dataset
    return indices_random[n_val:], indices_random[:n_val]


train_ds, val_ds = split_indices(len(tensor_dataset), val_percent=0.2)

batch_size = 100
# Train sampler and data loader
train_sampler = sampler.SubsetRandomSampler(train_ds)
train_dl = DataLoader(dataset=tensor_dataset,
                      batch_size=batch_size,
                      sampler=train_sampler)

# Validation sampler and data loader
val_sampler = sampler.SubsetRandomSampler(val_ds)
val_dl = DataLoader(tensor_dataset,
                    batch_size,
                    sampler=val_sampler)

class MNISTModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input):
        input = input.reshape(-1, 784)
        out = self.linear(input)
        return out


model = MNISTModel(input_size=784, output_size=10)

# define function
learning_rate = 0.001
loss_fn = F.cross_entropy
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)


def loss_batch(model, loss_fn, xb, yb, opt=None, metric=None):

    # Calculate loss
    preds = model(xb)
    loss = loss_fn(preds, yb)

    if opt is not None:
        # Compute gradient
        loss.backward()
        # Update paramenters
        opt.step()
        # Reset gradient
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        # Compute metric
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result


def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        # Pass each batch through the model
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric)
                   for xb, yb in valid_dl]
        # Separate losses, counts and metrics
        losses, nums, metrics = zip(*results)
        # Total size of the dataset
        total = np.sum(nums)
        # Avg. loss across batches
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            # Avg. of metric across batches
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)


def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric=None):
    for epoch in range(epochs):
        # Training
        for xb, yb in train_dl:
            loss, _, _ = loss_batch(model, loss_fn, xb, yb, opt)

        # Evaluation
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        # Print progress
        if metric is None:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epochs, val_loss))
        else:
            print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'
                  .format(epoch + 1, epochs, val_loss, metric.__name__, val_metric))


if __name__ == '__main__':
    fit(10, model, F.cross_entropy, opt, train_dl, val_dl, accuracy)
