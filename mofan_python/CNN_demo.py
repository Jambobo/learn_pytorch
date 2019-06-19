import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
from torch.utils.data import sampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DOWNLOAD_MNIST = False
BATCH_SIZE = 50
LR = 0.01
EPOCH = 1

train_data = torchvision.datasets.MNIST(root='data/',
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST)

train_dl = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='data/', train=False)
val_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
val_y = test_data.test_labels[:2000]





