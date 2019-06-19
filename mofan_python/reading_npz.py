import numpy as np

z = np.load("data/embedding.npz")
print(z)

y = np.load("data/embedding.npy")
print(y)
print(y.shape)

train_ds = np.load("data/train.json")
print(train_ds.shape)