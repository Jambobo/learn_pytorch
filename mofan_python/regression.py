import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

x, y = Variable(x), Variable(y)

# dataset
train_ds = TensorDataset(x, y)
batch_size = 10
train_dl = DataLoader(train_ds, batch_size, shuffle=False)


# official steps--define network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output): # num_input, num_neral, num_output
        super(Net, self).__init__()

        #difine layer
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))  #activation
        pred = self.predict(x)
        return pred


net = Net(n_feature=1, n_hidden=10, n_output=1)

# loss function
# mse used in regression problem
loss_fn = F.mse_loss

# optimizer function
opt = torch.optim.SGD(net.parameters(), lr=0.2)

# Draw
plt.ion()
plt.show()


def train():
    for epoch in range(200):
        # use self.forward here
        pred = net(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # make the result visualization
        if epoch % 5 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), pred.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

def save():
    torch.save(net, 'net/net.pt')           # entire net
    torch.save(net.state_dict(), 'net/net_params.pt')    # only parameters


def restore_net():
    net2 = torch.load('net/net.pt')
    print(net2)
    print(net2.state_dict())


def restore_params():
    # net3 = torch.nn.Sequential(     # require to be the same as Net
    #     torch.nn.Linear(1, 10),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(10, 1),
    # )

    net3 = Net(1, 10, 1)

    net3.load_state_dict(torch.load('net/net_params.pt'))

    print(net3)


if __name__ == '__main__':
    # train()
    # save()
    restore_net()
    restore_params()







