import torch
from torch.optim import Optimizer
import tilt

x = torch.zeros(2)
objective = lambda x: x**2

optimizer = tilt.tilt(lr=0.1, mu=0.9, tau=0.999, beta=0.9)

for i in range(10000):
    pass

