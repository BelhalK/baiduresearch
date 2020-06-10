import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
import math
import numpy as np

class SAGDSparse(Optimizer):
    def __init__(self, params, lr=required, momentum=0, noise =0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, noise = noise,dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SAGDSparse, self).__init__(params, defaults)
        self.noise_groups = []
        for group in self.param_groups:
            noise_group = []
            for p in group['params']:
                noise = torch.zeros(p.size()).to(p.device)
                noise_group.append(noise)
            self.noise_groups.append(noise_group)


    def __setstate__(self, state):
        super(SAGDSparse, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def get_grad(self):
        self.grad_groups = []
        for group in self.param_groups:
            grad_group = []
            for p in group['params']:
                d_p = p.grad.data
                grad_group.append(d_p.clone())
            self.grad_groups.append(grad_group)

        return self.grad_groups

    def step(self, grad1_groups, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, noise_group, grad1_group in zip(self.param_groups, self.noise_groups, grad1_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']


            for p, noise, grad1 in zip(group['params'], noise_group, grad1_group):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                scale = torch.abs(torch.max(d_p) - torch.min(d_p))
                noi = group['noise'] * scale
                noi = noi.float()

                diff = d_p - grad1

                gamma = torch.normal(torch.zeros(1), torch.ones(1)*noi)
                delta = torch.normal(torch.zeros(1), torch.ones(1)*noi*2)
                threshold = gamma + delta
                noise.normal_(0, noi)
                if torch.norm(diff).item() > threshold.item():
                    d_p = d_p.add_(noise)
                else:
                    d_p = grad1
#                scale = torch.abs(torch.max(d_p) - torch.min(d_p))
#                noi = group['noise'] * scale
#                noise.normal_(0, noi)
#                d_p = d_p.add_(noise)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

#                scale = torch.abs(torch.max(d_p) - torch.min(d_p))
#                print(scale)

                # oi = group['noise'] * scale
                # noise.normal_(0, noi)
                # noise = torch.normal( torch.zeros(d_p.size()), noi*torch.ones(d_p.size())).to(device)
                # d_p = d_p.add_(noise)
                p.data.add_(-group['lr'], d_p)
        return loss