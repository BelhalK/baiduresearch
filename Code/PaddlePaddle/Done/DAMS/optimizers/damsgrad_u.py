import math
import paddle
import numpy as np


class DAMSGrad():
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
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
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
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        self.params = params
        self.state = dict(lr=lr,exp_avg = m, exp_avg_sq=v, 
                        v_hat = v_hat, betas=betas, weight_decay=weight_decay,
                        num_round = num_round, eps = eps, lambda0 = lambda0, LAMB=LAMB,amsgrad = amsgrad, step = 0)
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, 
            v_hat=v_hat, m=m, v=v, num_round=num_round, LAMB = LAMB, lambda0=lambda0
        )

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        for group in self.param_groups:
            for p in group["params"]:
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
                    state["exp_avg"] = paddle.zeros_like(
                        p, memory_format=paddle.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = paddle.zeros_like(
                        p, memory_format=paddle.preserve_format
                    )
                    # The true adaptive learning rate used for update, value should be changed outside of the optimizer
                    state["adp_u"] = paddle.zeros_like(
                        p, memory_format=paddle.preserve_format
                    )
                    state["old_max_exp_avg_sq"] = paddle.zeros_like(
                        p, memory_format=paddle.preserve_format
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["old_max_exp_avg_sq"] = state["max_exp_avg_sq"]
                        state["max_exp_avg_sq"] = paddle.zeros_like(
                            p, memory_format=paddle.preserve_format
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg = beta1*exp_avg + (1 - beta1)*grad
                exp_avg_sq = beta2*exp_avg_sq + (1 - beta2)*paddle.multiply(grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    paddle.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                #      denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                else:
                    # denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    # use adpative learning rate consensused outside of the optimizer to update
                    
                # keep the true adaptive learning positive, line 10
                denom = np.maximum(math.sqrt(state["adp_u"],0.001))
                step_size = group["lr"] / bias_correction1
                # parameter update, line 11
                p.addcdiv_(exp_avg, denom, value=-step_size)
                # update u when parameter is finished, line 12
                state["adp_u"] =  state["adp_u"] + max_exp_avg_sq - state["old_max_exp_avg_sq"]


        return loss
