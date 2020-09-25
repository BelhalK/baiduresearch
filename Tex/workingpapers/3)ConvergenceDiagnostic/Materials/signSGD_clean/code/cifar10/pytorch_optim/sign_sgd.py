import torch
from .optimizer import Optimizer, required


class sign_SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum)
    with convergence diagnostic based on running sum of inner product of
    consecutive gradients.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 sim='cosine'):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(sign_SGD, self).__init__(params, defaults)

        # SGD_Diagnostic
        self.old_dp_loss = []
        for group in self.param_groups:
            g_loss_ls = []
            for p in group['params']:
                p_loss = p + 0
                p_loss.zero_()
                g_loss_ls.append(p_loss)
            self.old_dp_loss.append(g_loss_ls)
        if sim == "ip":
            def sim(x, y, z): return x
        elif sim == "cosine":
            def sim(x, y, z):
                return x / max(torch.sqrt(y) * torch.sqrt(z), 1e-8)
        else:
            raise ValueError("Invalid scaling argument: {}".format(sim))
        self.sim = sim

    def __setstate__(self, state):
        super(sign_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = torch.sign(p.grad.data)
                dp_loss = d_p + 0 # for convergence  diagnostic

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
                dp_opt = d_p + 0

                p.data.add_(-group['lr'], d_p)

                # SGD_Diagnostic
                ip[0] += torch.mm(dp_loss.view(1, -1), self.old_dp_loss[i][j].view(-1, 1))
                dpsq[0] += torch.sum(dp_loss**2)
                old_dpsq[0] += torch.sum(self.old_dp_loss[i][j]**2)
                self.old_dp_loss[i][j] = dp_loss + 0
                ggs[0] += torch.norm(dp_loss)

        # normalize inner product
        ip[0] = self.sim(ip[0], dpsq[0], old_dpsq[0])
        diag_stats = {'ip_loss':ip[0].data.cpu().numpy()[0][0], 'grad_loss':ggs[0].data.cpu().numpy().item()}

        return loss, diag_stats

    def change_momentum(self, new_momentum):
        if new_momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(new_momentum))

        for group in self.param_groups:
            group['momentum'] = new_momentum

            for p in group['params']:
                param_state = self.state[p]
                param_state['momentum_buffer'].zero_()
