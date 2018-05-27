import torch
from torch.optim import Optimizer

class tilt(Optimizer):
    """Implements tilt algorithm.
    It was suggested by Eric Drexler. 
    This code is adapted from the adadelta torch source.
    The update equations are as follows:
    Momentum and tilt hyperparameters:
    mu   =  momentum coefficient (typical value = ~~0.9)
    tau  =  tilt coefficient  (typical value = ~~0.999?)
    beta =  tilt weight

    Variables, updates:

    g    = actual gradient
    g*  :=  tau * g* + (1 - tau) g   =  tilt vector (smoothed gradient)
    g_2 := g + beta * g*             = effective (tilted) gradient
    v    :=  mu * v - (1 - mu) * g_2 =  velocity vector

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """


    def __init__(self, params, lr=0.1, mu=0.9, tau=0.999, beta=0.9):
        defaults = dict(mu=mu, tau=tau, beta=beta)
        super(tilt, self).__init__(params, defaults)
        self.mu = mu
        self.tau = tau
        self.beta = beta

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Tilt does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['smoothed_g'] = torch.zeros_like(p.data)
                    state['tilted_g'] = torch.zeros_like(p.data)
                    state['velocity'] = torch.zeros_like(p.data)

                smoothed_g, tilted_g, velocity = state['smoothed_g'], state['tilted_g'], state['velocity']
                #rho, eps = group['rho'], group['eps']

                state['step'] += 1

                #if group['weight_decay'] != 0:
                #grad = grad.add(group['weight_decay'], p.data)

                smoothed_g = self.tau * smoothed_g + (1 - self.tau) * grad
                tilted_g = grad + self.beta * smoothed_g
                velocity = self.mu * velocity - (1 - self.mu) * tilted_g
                delta = velocity * group['lr']
                p.data = torch.add(p.data, delta)
#                p.data.add(p.data, delta)

                # smoothed_g.mul_(rho).addcmul_(1 - rho, grad, grad)
                # std = smoothed_g.add(eps).sqrt_()
                # delta = tilted_g.add(eps).sqrt_().div_(std).mul_(grad)
                # p.data.add_(-group['lr'], delta)
                # tilted_g.mul_(rho).addcmul_(1 - rho, delta, delta)

        return loss



    
