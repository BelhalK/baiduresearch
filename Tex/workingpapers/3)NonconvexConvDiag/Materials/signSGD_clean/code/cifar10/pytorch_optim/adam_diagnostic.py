import math
import torch
from .optimizer import Optimizer


class Adam_Diagnostic(Optimizer):
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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_Diagnostic, self).__init__(params, defaults)

        # SGD_Diagnostic
        self.old_dp_loss = []
        self.old_dp_opt = []
        for group in self.param_groups:
            g_loss_ls = []
            g_opt_ls = []
            for p in group['params']:
                p_loss = p + 0
                p_loss.zero_()
                p_opt = p + 0
                p_opt.zero_()
                g_loss_ls.append(p_loss)
                g_opt_ls.append(p_opt)
            self.old_dp_loss.append(g_loss_ls)
            self.old_dp_opt.append(g_opt_ls)
        def ip_sim(x, y, z):
            return x / max(torch.sqrt(y) * torch.sqrt(z), 1e-8)
            # num = torch.mm(x.view(1, -1), y.view(-1, 1)) denom = max(torch.norm(x) * torch.norm(y), 1e-8)
            # return num / denom
        self.ip_sim = ip_sim

    def __setstate__(self, state):
        super(Adam_Diagnostic, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        ip = [0.0, 0.0] # Loss(0), Opt(1)
        dpsq = [0.0, 0.0]
        old_dpsq = [0.0, 0.0]
        ggs = [0.0, 0.0] # grads
        pps = 0.0 # params
        if closure is not None:
            loss = closure()

        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                dp_loss = grad + 0 # for convergence  diagnostic
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                dp_opt = exp_avg / denom + 0

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # SGD_Diagnostic
                ip[0] += torch.mm(dp_loss.view(1, -1), self.old_dp_loss[i][j].view(-1, 1))
                ip[1] += torch.mm(dp_opt.view(1, -1), self.old_dp_opt[i][j].view(-1, 1))
                dpsq[0] += torch.sum(dp_loss**2)
                dpsq[1] += torch.sum(dp_opt**2)
                old_dpsq[0] += torch.sum(self.old_dp_loss[i][j]**2)
                old_dpsq[1] += torch.sum(self.old_dp_opt[i][j]**2)
                self.old_dp_loss[i][j] = dp_loss + 0
                self.old_dp_opt[i][j] = dp_opt + 0
                ggs[0] += torch.norm(dp_loss)
                ggs[1] += torch.norm(dp_opt)
                pps += torch.norm(p)

        ip[0] = self.ip_sim(ip[0], dpsq[0], old_dpsq[0])
        ip[1] = self.ip_sim(ip[1], dpsq[1], old_dpsq[1])
        diag_stats = {'ip_loss':ip[0].data.cpu().numpy()[0][0], 'grad_loss':ggs[0].data.cpu().numpy().item(),
                      'ip_opt':ip[1].data.cpu().numpy()[0][0], 'grad_opt':ggs[1].data.cpu().numpy().item(),
                      'param':pps.data.cpu().numpy().item()}

        return loss, diag_stats
