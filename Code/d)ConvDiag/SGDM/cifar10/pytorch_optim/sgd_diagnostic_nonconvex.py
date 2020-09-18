import torch
from .optimizer import Optimizer, required

class SGD_Diagnostic_Nonconvex(Optimizer):
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
        super(SGD_Diagnostic_Nonconvex, self).__init__(params, defaults)

        # SGD_Diagnostic_Nonconvex
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
        super(SGD_Diagnostic_Nonconvex, self).__setstate__(state)
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
                d_p = p.grad.data
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

                # SGD_Diagnostic_Nonconvex
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
