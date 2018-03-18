import torch
from torch.optim import Optimizer
from torch import autograd
import matplotlib.pyplot as plt
import tilt


# Brief explanation of how torch computes its gradients
# Say we have a function z(x, y, z)
# In order to calculate dz/dx|_{x=x, y=y, z=z}, we have to
# initialise x, y, z,
# out = f(x, y, z)
# out.backwards ## This step does the backpropogation
# then the value of x.grad is dz/dx_{x=x, y=y, z=z},
# and the value of y.grad is dz/dy_{x=x, y=y, z=z},
# and so on 

x = autograd.Variable(torch.FloatTensor([0, 3]).resize_(2, 1), requires_grad=True)

def f(z):
    T = autograd.Variable(torch.FloatTensor([[2, 1], [1, 2]]))
    intermediate = torch.matmul(z.t(), T)
    return torch.matmul(intermediate, z).sum()


optimizer = tilt.tilt([{'params': x, 'lr':2}], lr=0.1, mu=0.9, tau=0.999, beta=0.9)
param_history = []
#optimizer = torch.optim.Adam([{'params': x, 'lr':0.1}])

for i in range(100):
    param_history.append((float(x.data[0]), float(x.data[1])))
    x.grad = None
    y = f(x)
    y.backward()
    #print('gradient of x is {} while x itself is {}'.format(x.grad, x))
    optimizer.step()

x = autograd.Variable(torch.FloatTensor([0, 3]).resize_(2, 1), requires_grad=True)    
optimizer = torch.optim.Adam([{'params': x, 'lr':0.2, 'betas':[0.9, 0.99]}])
param_history_torch = []
#optimizer = torch.optim.Adam([{'params': x, 'lr':0.1}])
for i in range(100):
    param_history_torch.append((float(x.data[0]), float(x.data[1])))
    x.grad = None
    y = f(x)
    y.backward()
    #print('gradient of x is {} while x itself is {}'.format(x.grad, x))
    optimizer.step()

def pretty_plot_trajectory(list_of_xs_ys, color='b'):
    xs, ys = zip(*list_of_xs_ys)
    plt.plot(xs, ys, color='k', marker = 'o', markersize=12,
             linewidth=1, markerfacecolor=color, markeredgecolor=color)

    
pretty_plot_trajectory(param_history)
pretty_plot_trajectory(param_history_torch, color='r')
plt.show()
