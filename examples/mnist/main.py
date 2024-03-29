from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from tilt import tilt
import itertools

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
if args.cuda:
    model.cuda()



def train(epoch, optimizer):
    model.train()
    avg_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        avg_loss += loss.data[0]
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    return avg_loss / len(train_loader)
        

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


#for index, lr in enumerate(np.logspace(-0.5, -5, 10)):
#mus = 1 - np.logspace(0, -10, 2)
#taus = 1 - np.logspace(-1, -10, 2)

# mus = [0.5, 0.0]         
# taus = [0.999, 0]        
# betas = [0, 0.1, 0.3, 1] 
mus = [0.5, 0.0, 0.9]
taus = [0.0, 0.99, 0.999]
betas = [0.0, 1.0, 0.2]

best_tilt_test_acc = 0
best_SGD_test_acc = 0

SGD_train = []
SGD_test = []
tilt_train = []
tilt_test = []

best_SGD_train = []
best_SGD_test = []
best_tilt_train = []
best_tilt_test = []

all_tilt_results = []

for index, lr in enumerate([0.01, 0.001, 0.0001]):
    print('Learning rate is {}'.format(lr))
    
    for index1, (mu, tau, beta) in enumerate(itertools.product(mus, taus, betas)):
        tilt_train = []
        tilt_test = []
        model = Net()
        if args.cuda: model.cuda()
        optimizer = tilt([{'params': model.parameters(), 'lr': lr}])
        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch, optimizer)
            print(train_loss)
            if np.isnan(train_loss):
                break
            tilt_train.append(train_loss)
            tilt_test.append(test())
        all_tilt_results.append(((mu, tau, beta), tilt_test[-1]))
        if tilt_test[-1] > best_tilt_test_acc:
            best_tilt_train = tilt_train
            best_tilt_test = tilt_test
            best_tilt_test_acc = tilt_test[-1]
            
    model = Net()
    if args.cuda: model.cuda()    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    SGD_train = []
    SGD_test = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch, optimizer)
        if np.isnan(train_loss):
            break
        SGD_train.append(train_loss)
        SGD_test.append(test())
    if SGD_test[-1] > best_SGD_test_acc:
        best_SGD_train = SGD_train
        best_SGD_test = SGD_test
        best_SGD_test_acc = SGD_test[-1]

plt.plot(range(args.epochs), best_SGD_train, color='k', marker = 'o', markersize=12,
               linewidth=1, markerfacecolor='k', markeredgecolor='k', label='SGD Train')

plt.plot(range(args.epochs), best_SGD_test, color='k', marker = '+', markersize=12,
               linewidth=1, markerfacecolor='k', markeredgecolor='k', label='SGD Test')

plt.plot(range(args.epochs), best_tilt_train, color='r', marker = 'o', markersize=12,
         linewidth=1, markerfacecolor='r', markeredgecolor='r', label='Tilt Train')

plt.plot(range(args.epochs), best_tilt_test, color='r', marker = '+', markersize=12,
         linewidth=1, markerfacecolor='r', markeredgecolor='r', label='Tilt Test')
plt.legend()
plt.savefig('saved.png', dpi=1200)
print([str(item) + '\t' for item in all_tilt_results])
