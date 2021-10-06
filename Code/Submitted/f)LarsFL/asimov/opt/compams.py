import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
import time
import os
from .sparsification_dist import sign_grad, compute_topk
import collections
import sys
from .compressor import *

import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class CompAMS(Optimizer):

    def __init__(self, params, args, log_writer, **kwargs):

        lr = args.lr
        momentum = args.momentum
        weight_decay = args.weight_decay
        betas=(0.9, 0.999)
        eps=1e-8
        amsgrad=True
        compression_buffer = args.compress
        all_reduce = args.all_reduce
        local_rank = args.local_rank
        gpus_per_machine = args.gpus_per_machine

        self.compression_buffer = compression_buffer
        self.all_reduce = all_reduce
        self.signum = args.signum
        self.log_writer = log_writer

        self.args = args

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # defaults = dict(lr=lr, momentum=momentum,
        #                 weight_decay=weight_decay)

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum=momentum, amsgrad=amsgrad
        )
        
        super(CompAMS, self).__init__(params, defaults)

        self.MB = 1024 * 1024
        self.bucket_size = 100 * self.MB


        self.compressor = compressor(using_cuda = True, local_rank = local_rank)
        self.local_rank = local_rank
        self.global_rank = dist.get_rank()
        self.local_dst_in_global = self.global_rank - self.local_rank

        self.inter_node_group = []
        self.nodes = dist.get_world_size() // gpus_per_machine

        self.intra_node_group_list = []

        for index in range(self.nodes):
            # set inter_node_group
            self.inter_node_group.append(0 + index * gpus_per_machine)
            # set all intra_node_group
            intra_node_group_temp = []
            for intra_index in range(gpus_per_machine):
                intra_node_group_temp.append(intra_index + index * gpus_per_machine)
            intra_node_group_temp = dist.new_group(intra_node_group_temp)
            self.intra_node_group_list.append(intra_node_group_temp)

            if self.local_dst_in_global == 0 + index * gpus_per_machine:
                self.nodes_rank = index


        #self.intra_node_list = self.intra_node_group
        self.inter_node_list = self.inter_node_group
        self.inter_node_group_list = []
        for index in range(len(self.inter_node_list)):
            if index != 0:
                temp = dist.new_group([self.inter_node_list[0],self.inter_node_list[index]])
                self.inter_node_group_list.append(temp)
        self.all_gpu = dist.new_group()

        self.all_inter_node_group = dist.new_group(self.inter_node_list)

        if dist.get_rank() == 0 or dist.get_rank() == 8:
            print('nodes', self.nodes)
            print('intra_node_group_list',self.intra_node_group_list)
            print('inter_node_group',self.inter_node_group_list)
            print('all_inter_node_group', self.inter_node_list)

    def __setstate__(self, state):
        super(CompAMS, self).__setstate__(state)


    def step(self, closure=None):

        args = self.args

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            all_grads = []
            i=0
            for p in group['params']:
                i +=1
                # print(i)

                # ForkedPdb().set_trace()
                state = self.state[p]
                # ForkedPdb().set_trace()
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # The true adaptive learning rate used for update, value should be changed outside of the optimizer
                    state["adp_u"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["local_error"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                
                
                exp_avg, exp_avg_sq, max_exp_avg_sq, local_error = state["exp_avg"], state["exp_avg_sq"], state["max_exp_avg_sq"], state["local_error"]
                    
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if p.grad is None:
                    continue

                d_p = p.grad.data
                if self.compression_buffer==False:
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    # signum
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    else:
                        buf = param_state['momentum_buffer']

                    buf.mul_(momentum).add_((1 - momentum),d_p)
                    d_p.copy_(buf)

                all_grads.append(d_p)

                # dev_grads_buckets = _take_tensors(all_grads, self.bucket_size)
                
                # len(all_grads) = 62
                # all_grads[0].shape = torch.Size([64, 3, 7, 7])
                
                # len(dev_grads) = 62
                # dev_grads[0].shape = torch.Size([64, 3, 7, 7])

                # shape of dev_grads_buckets = ?
                
                # for dev_grads in dev_grads_buckets:
                


                # d_p_new = _flatten_dense_tensors(d_p)
                d_p_new = d_p

                if self.all_reduce:
                    dist.all_reduce(d_p_new) #self.all_gpu, group = 0
                    if self.signum:
                        d_p_new = torch.sign(d_p_new)
                elif self.signum:
                    if self.nodes > 1:
                        if self.compression_buffer:
                            d_p_new, tensor_size = self.compressor.compress(d_p_new)
                        else:
                            # d_p_new = torch.sign(d_p_new)
                            if args.method == 'topk':
                                d_p_new_comp = compute_topk(d_p_new + local_error,max(args.sparsity,1/d_p_new.numel())).float()
                                local_error.to(p.device).add_(d_p_new.to(p.device)-d_p_new_comp.to(p.device))
                                d_p_new = d_p_new_comp.to(p.device)
                            
                            if args.method == 'sign':
                                d_p_new_comp = sign_grad(d_p_new + local_error).float()
                                local_error.to(p.device).add_(d_p_new.to(p.device)-d_p_new_comp.to(p.device))
                                d_p_new = d_p_new_comp.to(p.device)

                            if args.method == 'full':
                                d_p_new = d_p_new
                            
                        if self.local_rank == 0:
                            if dist.get_rank() == 0:
                                d_p_new_list = []
                                for index, inter_node_group in enumerate(self.inter_node_group_list):
                                    d_p_temp = d_p_new.clone()
                                    dist.broadcast(d_p_temp.to(p.device), self.inter_node_list[index + 1], group = inter_node_group)
                                    d_p_new_list.append(d_p_temp)
                            else:
                                dist.broadcast(d_p_new, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1])                                
                                dist.barrier(group = self.all_inter_node_group)

                            if dist.get_rank() == 0:
                                if self.compression_buffer:
                                    d_p_new_list.append(d_p_new) #count itself
                                    d_p_new = self.compressor.majority_vote(d_p_new_list)
                                else:
                                    for d_p_temp in d_p_new_list:
                                        d_p_new.add_(d_p_temp)
                                    d_p_new = d_p_new / self.nodes
                                dist.barrier(group = self.all_inter_node_group)
                            dist.broadcast(d_p_new, 0, group = self.all_inter_node_group)

                        if self.compression_buffer:
                            d_p_new = self.compressor.uncompress(d_p_new, tensor_size)
                else:
                    print('You can not run without signum or all_reduce')

                #unflatten
                # dev_grads_new = _unflatten_dense_tensors(d_p_new,d_p)
                for grad, reduced in zip(d_p, d_p_new):
                    grad.copy_(reduced)
                    # print(grad.shape)
            
                # #LARC saving
                # self.layer_adaptive_lr = []
                # layer_index = 0
                # laryer_saving = [1,2,3,23,49,87] #conv1.weight(no bias), bn1.weight, layer1.1.conv1.weight, layer2.1.conv1.weight, layer3.1.conv1.weight, layer4.1.conv1.weight
                # ###
                # for p in group['params']:
                # state = self.state[p]
                # # ForkedPdb().set_trace()
                
                # # State initialization
                # if len(state) == 1:
                #     state["step"] = 0
                #     # Exponential moving average of gradient values
                #     state["exp_avg"] = torch.zeros_like(
                #         p, memory_format=torch.preserve_format
                #     )
                #     # Exponential moving average of squared gradient values
                #     state["exp_avg_sq"] = torch.zeros_like(
                #         p, memory_format=torch.preserve_format
                #     )
                #     state["max_exp_avg_sq"] = torch.zeros_like(
                #         p, memory_format=torch.preserve_format
                #     )
                #     # The true adaptive learning rate used for update, value should be changed outside of the optimizer
                #     state["adp_u"] = torch.zeros_like(
                #         p, memory_format=torch.preserve_format
                #     )
                #     state["local_error"] = torch.zeros_like(
                #         p, memory_format=torch.preserve_format)
                
                
                # exp_avg, exp_avg_sq, max_exp_avg_sq, local_error = state["exp_avg"], state["exp_avg_sq"], state["max_exp_avg_sq"], state["local_error"]
                    
                # beta1, beta2 = group["betas"]

                # state["step"] += 1
                # bias_correction1 = 1 - beta1 ** state["step"]
                # bias_correction2 = 1 - beta2 ** state["step"]

                # layer_index += 1
                ###
                # '''
                # LARC
                # This part of code was originally forked from (https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py)
                # ''' 
                # if args.larc_enable:
                #     trust_coefficient = args.larc_trust_coefficient
                #     clip = args.larc_clip
                #     eps = args.larc_eps
                #     param_norm = torch.norm(p.data)
                #     grad_norm = torch.norm(p.grad.data)
                #     if param_norm != 0 and grad_norm != 0:
                #         # calculate adaptive lr + weight decay
                #         adaptive_lr = trust_coefficient * (param_norm) / (grad_norm + param_norm * weight_decay + eps)

                #         #add adaptive lr saving
                #         if layer_index in laryer_saving:
                #             self.layer_adaptive_lr.append(adaptive_lr)

                #         # clip learning rate for LARC
                #         if clip:
                #             # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                #             adaptive_lr = min(adaptive_lr/group['lr'], 1)

                #         else:
                #             adaptive_lr = adaptive_lr/group['lr']

                #         p.grad.data *= adaptive_lr
                # ###


                # if self.compression_buffer: #This part of code is temporary
                #     if weight_decay != 0:
                #         p.grad.data.add_(weight_decay, p.data)
                # p.data.add_(-group['lr'], p.grad.data)

                # Decay the first and second moment running average coefficient
                exp_avg.to(p.device).mul_(beta1).add_(p.grad.data.to(p.device), alpha=1 - beta1)
                exp_avg_sq.to(p.device).mul_(beta2).addcmul_(p.grad.data.to(p.device), p.grad.data.to(p.device), value=1 - beta2)
                
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                            
                sqv = max_exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group["lr"]              

                p.data.addcdiv_(exp_avg.to(p.device), sqv.to(p.device), value=-step_size)
                
                # p.data.add_(-group['lr'], p.grad.data)
                # else:
                #     denom = np.maximum(torch.sqrt(state["adp_u"]),0.001)
                #     step_size = group["lr"] / bias_correction1
                #     p.addcdiv_(exp_avg.to(p.device), denom, value=-step_size)

        return loss

    def get_v(self):
        
        vv=collections.OrderedDict()
        
        for group in self.param_groups:
                for i,p in enumerate(group["params"]):
                    # if p.grad is None:
                    #     continue
                    
                    # grad = p.grad
                    # if grad.is_sparse:
                    #     raise RuntimeError(
                    #         "Adam does not support sparse gradients, please consider SparseAdam instead"
                    #     )
                    
                    state = self.state[p]
                    vv[i] = state['max_exp_avg_sq']
                    
        return vv
                
    def get_m(self):
        
        mm=collections.OrderedDict()
        
        for group in self.param_groups:
                for i,p in enumerate(group["params"]):
                    # if p.grad is None:
                    #     continue
                    # grad = p.grad
                    # if grad.is_sparse:
                    #     raise RuntimeError(
                    #         "Adam does not support sparse gradients, please consider SparseAdam instead"
                    #     )
    
                    state = self.state[p]                    
                    mm[i] = state['exp_avg']
                    
        return mm 


