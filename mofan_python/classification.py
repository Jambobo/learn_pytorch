import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# fake data
n_data = torch.ones(100, 2)         # basic data

# torch.normal(mean, var)
# mean = 2, var = 1
x0 = torch.normal(2*n_data, 1)      # x0 data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # y0 data (tensor), shape=(100, 1)

# mean = 2, var = 1
x1 = torch.normal(-2*n_data, 1)     # x1 data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # y1 data (tensor), shape=(100, 1)

# combine the matrix according to the 0-dim
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)  # label must be LongTensor

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

plt.ion()   # draw
plt.show()


# define the nerwork
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        out = self.out(x)
        return out

# method 2  quick build
Net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 15),
    torch.nn.ReLU(),
    torch.nn.Liner(15, 2),
)


net = Net(n_feature=2, n_hidden=15, n_output=2)

# F.cross_entropy used in classification problems
loss_fn = F.cross_entropy
opt = torch.optim.SGD(net.parameters(), lr=0.02)


def train():
    for epoch in range(100):
        out = net(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # print(loss.data.numpy())

        if epoch % 2 == 0:
            plt.cla()
            # need a sofmax to get probability
            prediction = torch.max(F.softmax(out, dim=0), 1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn' )
            accuracy = sum(pred_y == target_y)/200
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

        plt.ioff()  # stop draw
        plt.show()


if __name__ == '__main__':
    train()


