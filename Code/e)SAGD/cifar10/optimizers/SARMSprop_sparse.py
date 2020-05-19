import torch
from torch.optim import Optimizer


class SARMSpropSparse(Optimizer):
    """Implements RMSprop algorithm.
    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.
    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, noise =0, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha,  noise = noise, eps=eps, centered=centered, weight_decay=weight_decay)
        super(SARMSpropSparse, self).__init__(params, defaults)
        self.noise_groups = []
        for group in self.param_groups:
            noise_group = []
            for p in group['params']:
                noise = torch.zeros(p.size()).to(p.device)
                noise_group.append(noise)
            self.noise_groups.append(noise_group)
        
        
        

    def __setstate__(self, state):
        super(SARMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)
            
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
            
            for p, noise, grad1 in zip(group['params'], noise_group, grad1_group):
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                scale = torch.abs(torch.max(d_p) - torch.min(d_p))
                noi = group['noise'] * scale
                noi = max(noi, 1e-5)
                noise.normal_(0, noi)

                diff = d_p - grad1
                gamma = torch.normal(torch.zeros(1), torch.ones(1)*noi)
                delta = torch.normal(torch.zeros(1), torch.ones(1)*noi*2)
                threshold = gamma + delta
                
                if torch.norm(diff).item() > threshold.item():
                    d_p = d_p.add_(noise)
                else:
                    d_p = grad1

                grad = d_p
                
#                grad = p.grad.data
#                
#                scale = torch.abs(torch.max(grad) - torch.min(grad))
#                
#                noi = group['noise'] * scale
#                
#                noise.normal_(0, noi)
#                
#                grad = grad.add_(noise)
                
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss