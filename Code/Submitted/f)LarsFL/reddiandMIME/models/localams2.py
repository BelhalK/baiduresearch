# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:23:00 2020

@author: bdfzl
"""

import math
import torch
from torch.optim import Optimizer
import numpy as np
import pdb
import collections

class LocalAMSGrad(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        v_hat,
        v,
        m,
        num_round,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=True,
        LAMB=False,
        lambda0=0.01
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, 
            v_hat=v_hat, m=m, v=v, num_round=num_round, LAMB = LAMB, lambda0=lambda0
        )
        super(LocalAMSGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LocalAMSGrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i,p in enumerate(group["params"]):
#                print(p.grad)
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = group["m"][i]
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = group["v"][i]
                    # The true adaptive learning rate used for update, value should be changed outside of the optimizer
                    state["adp_u"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = group["v_hat"][i]
                    
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    if group["num_round"]==0:
                        sqv = exp_avg_sq.sqrt().add_(group['eps'])
                    else:                            
                        sqv = max_exp_avg_sq.sqrt().add_(group['eps'])
                    step_size = group["lr"]
                    lambda0 = group["lambda0"]
                    if group["LAMB"]:
                        r= exp_avg/sqv
                        scale=torch.norm(p)/torch.norm(r+lambda0*p)                           
                        p.addcmul_(scale, r+lambda0*p, value=-step_size)
                    else:                    
                        p.addcdiv_(exp_avg, sqv, value=-step_size)
                else:
                    denom = np.maximum(torch.sqrt(state["adp_u"]),0.001)
                    step_size = group["lr"] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def get_v(self):
        
        vv=collections.OrderedDict()
        
        for group in self.param_groups:
                for i,p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
    
                    state = self.state[p]
                    vv[i] = state['exp_avg_sq']
                    
        return vv
                
    def get_m(self):
        
        mm=collections.OrderedDict()
        
        for group in self.param_groups:
                for i,p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
    
                    state = self.state[p]                    
                    mm[i] = state['exp_avg']
                    
        return mm  