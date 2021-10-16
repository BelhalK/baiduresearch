import math
import paddle
import numpy as np
import pdb
import collections


class LocalSGD():

    def __init__(self, params, LAMB, lambda0, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.params = params
        self.state = dict(weight_decay=weight_decay,
                        dampening = dampening, nesterov = nesterov)

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

    
        weight_decay = self.state['weight_decay']
        momentum = self.state['momentum']
        dampening = self.state['dampening']
        nesterov = self.state['nesterov']

        for p in self.params:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = paddle.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf = paddle.multiply(buf,momentum) +(1 - dampening)*d_p

                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            
            if self.state["LAMB"]:
                m = d_p+self.state['lambda0']*p
                A = m*paddle.norm(p)/paddle.norm(m)
            else:
                A =d_p

            p.data.add_(-self.state['lr'], A)

        return loss
