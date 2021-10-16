import math
import paddle
import numpy as np
import pdb
import collections


class LocalLAMB():
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
        
        self.params = params
        self.state = dict(lr=lr,exp_avg = m, exp_avg_sq=v, 
                        v_hat = v_hat, betas=betas, weight_decay=weight_decay,
                        num_round = num_round, eps = eps, lambda0 = lambda0, LAMB=LAMB,amsgrad = amsgrad, step = 0)
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, 
            v_hat=v_hat, m=m, v=v, num_round=num_round, LAMB = LAMB, lambda0=lambda0
        )

    # def __setstate__(self, state):
    #     super(LocalAMSGrad, self).__setstate__(state)
    #     for group in self.param_groups:
    #         group.setdefault("amsgrad", False)

    def step(self):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        for i,p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad

            # if not self.state["adp_u"].has_key(i):
            # self.state["adp_u"][i] = paddle.zeros_like(p)
            self.state["adp_u"] = paddle.zeros_like(p)

            exp_avg, exp_avg_sq = self.state["exp_avg"][i], self.state["exp_avg_sq"][i]
            if self.state["amsgrad"]:
                max_exp_avg_sq = self.state["v_hat"][i]
                
            beta1, beta2 = self.state["betas"]

            self.state["step"] += 1
            bias_correction1 = 1 - beta1 ** self.state["step"]
            bias_correction2 = 1 - beta2 ** self.state["step"]

            if self.state["weight_decay"] != 0:
                grad = grad.add(p*self.state["weight_decay"])

            # Decay the first and second moment running average coefficient
            exp_avg = beta1*exp_avg + (1 - beta1)*grad
            exp_avg_sq = beta2*exp_avg_sq + (1 - beta2)*paddle.multiply(grad, grad)
            
            if self.state["amsgrad"]:
                # Maintains the maximum of all 2nd moment running avg. till now
                if self.state["num_round"]==0:
                    sqv = exp_avg_sq.sqrt()+self.state['eps']
                else:                            
                    sqv = max_exp_avg_sq.sqrt()+self.state['eps']
                step_size = self.state["lr"]
                lambda0 = self.state["lambda0"]
                if self.state["LAMB"]:
                    r=exp_avg/sqv
                    scale=paddle.norm(p)/paddle.norm(r+lambda0*p)                           
                    p = p - step_size*paddle.multiply(scale, r+lambda0*p)
                else:                    
                    p = p - step_size*paddle.divide(exp_avg, sqv)
            else:
                denom = np.maximum(paddle.sqrt(self.state["adp_u"]),0.001)
                step_size = self.state["lr"] / bias_correction1
                p = p - step_size*paddle.divide(exp_avg, denom)
        return loss

    def get_v(self):
        
        vv=collections.OrderedDict()
        
        for i,p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad

            # state = self.state[p]
            state = self.state
            vv[i] = state['exp_avg_sq']
                    
        return vv
                
    def get_m(self):
        
        mm=collections.OrderedDict()
        
        for i,p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad

            # state = self.state[p]
            state = self.state                   
            mm[i] = state['exp_avg']
            
        return mm  