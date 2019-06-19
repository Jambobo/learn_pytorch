import torch
import numpy as np

np_data = np.array([[3,4],[5,9]])
torch_data = torch.from_numpy(np_data)
tensor2arry = torch_data.numpy()

print (
    '\nnumpy',np_data,
    '\ntorch',torch_data,
    '\ntensor2arry',tensor2arry,
)

np_data = np.array([[1, 2, 3],
                    [2, 4, 5],
                    [2, 3, 5],
                    [5, 6, 8]])

print(np_data.shape)