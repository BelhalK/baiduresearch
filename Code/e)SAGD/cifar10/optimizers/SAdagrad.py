import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required


class SAdagrad(Optimizer):

    def __init__(self, params, lr=required, lr_decay=0, noise=0, weight_decay=0, initial_accumulator_value=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))

        defaults = dict(lr=lr, lr_decay=lr_decay, noise = noise, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(SAdagrad, self).__init__(params, defaults)

        self.noise_groups = []
        for group in self.param_groups:
            noise_group = []
            for p in group['params']:
                noise = torch.zeros(p.size()).to(p.device)
                noise_group.append(noise)
            self.noise_groups.append(noise_group)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p.data, initial_accumulator_value)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group , noise_group in zip(self.param_groups, self.noise_groups):
            
            for p, noise in zip(group['params'], noise_group):
                if p.grad is None:
                    continue

                grad = p.grad.data
                
                state = self.state[p]

                state['step'] += 1
                
                scale = torch.abs(torch.max(grad) - torch.min(grad))
                
                noi = group['noise'] * scale
                noi = max(noi, 1e-5)
                noise.normal_(0, noi)
                
                grad = grad.add_(noise)
                

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum'].sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

        return loss

