# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.distributed as dist
from copy import deepcopy
import time

from pcode.utils.auxiliary import deepcopy_model

"""the frequency of communication"""

def configure_sync_scheme(args):
    args.local_steps = define_sync_freq(
        num_epochs=args.num_epochs,
        local_step=args.local_step,
        local_step_warmup_type=args.local_step_warmup_type,
        local_step_warmup_period=args.local_step_warmup_period,
        turn_on_local_step_from=args.turn_on_local_step_from,
        turn_off_local_step_from=args.turn_off_local_step_from,
        warmup_per_intervals=args.local_step_warmup_per_interval,
        lr_change_epochs=args.lr_change_epochs)


def define_sync_freq(
        num_epochs, local_step, local_step_warmup_type,
        local_step_warmup_period,
        turn_on_local_step_from, turn_off_local_step_from,
        warmup_per_intervals, lr_change_epochs):
    # TODO: should figure out a better sync scheme.
    # directly return a list of local steps.
    num_epochs = num_epochs + 2
    if local_step_warmup_period is None:
        local_step_warmup_period = local_step

    # we need local step warmup.
    # determine the local step warmup scheme.
    if local_step_warmup_type is None:
        tmp_steps = [local_step] * local_step_warmup_period
    elif 'exp' in local_step_warmup_type:
        log_local_step = int(np.log2(local_step_warmup_period))
        tmp_steps = [
            2 ** int(ind * log_local_step / local_step_warmup_period)
            for ind in range(1, 1 + local_step_warmup_period)
        ]
    elif 'linear' in local_step_warmup_type:
        tmp_steps = [
            max(1, int(ind * local_step / local_step_warmup_period))
            for ind in range(1, 1 + local_step_warmup_period)
        ]
    elif 'constant' in local_step_warmup_type:
        tmp_steps = [1] * local_step_warmup_period
    else:
        raise NotImplementedError

    if len(tmp_steps) > num_epochs:
        tmp_steps = tmp_steps[: num_epochs]

    # get lr_change_epochs.
    if lr_change_epochs is not None:
        lr_change_epochs = [int(x) for x in lr_change_epochs.split(',')]
        lr_change_epochs = [0] + lr_change_epochs + [num_epochs]
        lr_change_fromto_epochs = list(
            zip(lr_change_epochs[: -1], lr_change_epochs[1:])
        )

    # determine if we want to repeat the local step warmup or not.
    if not warmup_per_intervals:
        steps = []

        # add some specific operators.
        if lr_change_epochs is None:
            # allowed to use local step warmup
            steps = tmp_steps + [local_step] * (num_epochs - len(tmp_steps))
        else:
            # allowed to use local step warmup
            if turn_on_local_step_from is None and turn_off_local_step_from is None:
                return tmp_steps + [local_step] * (num_epochs - len(tmp_steps))

            # not allowed to use local step warmup.
            for from_ind, to_ind in lr_change_fromto_epochs:
                if turn_on_local_step_from is None and turn_off_local_step_from is not None:
                    if from_ind >= turn_off_local_step_from:
                        steps += [1] * (to_ind - from_ind)
                    else:
                        t = [local_step] * (to_ind - from_ind)
                        steps += t
                elif turn_on_local_step_from is not None and turn_off_local_step_from is None:
                    if from_ind >= turn_on_local_step_from:
                        t = [local_step] * (to_ind - from_ind)
                        steps += t
                    else:
                        steps += [1] * (to_ind - from_ind)
                elif turn_on_local_step_from is not None and turn_off_local_step_from is not None:
                    raise ValueError('not considering this case for the moment.')
    elif warmup_per_intervals:
        steps = []
        for from_ind, to_ind in lr_change_fromto_epochs:
            t = [local_step] * (to_ind - from_ind - len(tmp_steps))
            steps += tmp_steps + t
    else:
        raise NotImplementedError
    return steps


"""functions."""


def global_average(sum, count, group=None):
    def helper(array, group=None):
        array = torch.FloatTensor(array)
        if group is None:
            dist.all_reduce(array, op=dist.ReduceOp.SUM)
        else:
            dist.all_reduce(array, op=dist.ReduceOp.SUM, group=group)
        all_sum, all_count = array
        if all_count == 0:
            return 0
        else:
            return all_sum / all_count
    avg = helper([sum, count], group)
    return avg


def elementwise_min(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return tensor


""" Federated functions"""

def fedavg_aggregation(args, model_server, model_client, group, online_clients, optimizer, lambda_weight=None):
    """Aggregate gradients for federated learning.

    Each local model first gets the difference between current model and
    previous synchronized model, and then all-reduce these difference by SUM.

    """
    num_online_clients = len(online_clients) if 0 in online_clients else len(online_clients) + 1
    if (0 not in online_clients) and (args.graph.rank == 0):
        rank_weight = 0
    else:
        if lambda_weight is None:
            rank_weight =  args.num_samples_per_epoch / args.train_dataset_size
            # rank_weight =  1.0 / num_online_clients
        else:
            #TODO: This is experimental. Test it.
            rank_weight = lambda_weight * args.graph.n_nodes / num_online_clients

    for server_param, client_param in zip(model_server.parameters(), model_client.parameters()):
        # get model difference.
        client_param.grad.data = (server_param.data - client_param.data) * rank_weight
        # recover to old model.
        client_param.data = server_param.data
        # all reduce.
        dist.all_reduce(client_param.grad.data,  op=dist.ReduceOp.SUM, group=group)

    # apply gradient again.
    optimizer.step(
        apply_lr=False,
        scale=args.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=args.out_momentum,
    )

    # reassign model to old_model.
    model_server = deepcopy_model(args, model_client)

    return model_server


def set_online_clients(args):
    useable_ranks = args.graph.ranks
    ranks_shuffled = np.random.permutation(useable_ranks)
    online_clients = ranks_shuffled[:int(args.online_client_rate * len(useable_ranks))]

    online_clients = torch.IntTensor(online_clients)
    group = dist.new_group(args.graph.ranks)
    dist.broadcast(online_clients, src=0, group=group)
    return list(online_clients.numpy())

def distribute_model_server(model_server, group, src=0):
    """
    Distributing the model on server from source node to the group of process
    """
    for server_param in model_server.parameters():
        dist.broadcast(server_param.data, src=src, group=group)

    return model_server

def distribute_model_server_control(model_server, model_server_control, group, src=0):
    """
    Distributing the model on server from source node to the group of process
    #TODO: merge with distribute_model_server method
    """
    for server_param, server_control_param in zip(model_server.parameters(),model_server_control.parameters()):
        t = torch.stack([server_param.data, server_control_param.data])
        dist.broadcast(t, src=src, group=group)
        server_param.data = t[0]
        server_control_param.data = t[1]


    return model_server, model_server_control

def aggregate_models_virtual(args, model, group, online_clients):
    virtual_model = deepcopy_model(args, model)
    # rank_weight =  args.num_samples_per_epoch / args.train_dataset_size
    if (0 not in online_clients) and (args.graph.rank == 0):
        rank_weight = 0
    else:
        rank_weight =   1 / len(online_clients)
    for param in virtual_model.parameters():
        param.data *= rank_weight
        # all reduce.
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=group)
        # if or not averge the model.
        
    return virtual_model


def scaffold_aggregation(args, model_server, model_client, model_server_control, model_client_control,
                         group, online_clients, optimizer, lr, local_steps, lambda_weight=None):
    """Aggregate gradients for federated learning using SCAFFOLD.https://arxiv.org/abs/1910.06378"""
    model_client_control_copy = deepcopy(model_client_control)
    num_online_clients = len(online_clients) if 0 in online_clients else len(online_clients) + 1
    if (0 not in online_clients) and (args.graph.rank == 0):
        rank_weight = 0
    else:
        if lambda_weight is None:
            rank_weight =  1.0 / num_online_clients
        else:
            #TODO: This is experimental. Test it.
            rank_weight = lambda_weight * args.graph.n_nodes / num_online_clients
        # Update local control variates for online clients only
        for cccp, ccp, scp, cp, sp in zip(model_client_control_copy.parameters(), model_client_control.parameters(), model_server_control.parameters(), model_client.parameters(), model_server.parameters()):
            cccp.data = ccp.data - scp.data + (sp.data - cp.data)/(local_steps * lr)

    
    for cccp, ccp, scp, cp, sp in zip(model_client_control_copy.parameters(), model_client_control.parameters(), model_server_control.parameters(), model_client.parameters(), model_server.parameters()):
        # get model difference.
        cp.grad.data = (sp.data - cp.data) * rank_weight
        # recover to old model.
        cp.data = sp.data
        # Control variate change
        ccp.data = (ccp.data - cccp.data) * rank_weight

        t = torch.stack([cp.grad.data, ccp.data])
        # all reduce.
        # dist.all_reduce(t ,  op=dist.ReduceOp.SUM, group=group)
        gather_list = [torch.ones_like(t) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
        
        st = time.time()
        dist.gather(t, gather_list=gather_list, dst=0, group=group)
        if args.rounds_comm ==1 and args.graph.rank == 0:
                print("Size of the communication is {}".format(size_tensor(t)))
        delay_time = size_tensor(t) * args.comm_delay_coef
        delay_time += (delay_time/10.0) * np.random.randn()
        time.sleep(delay_time)
        args.comm_time[-1] += time.time() - st

        if args.graph.rank == 0:
            gather_list = gather_list if 0 in online_clients else gather_list[1:]
            d = torch.sum(torch.stack(gather_list,1), dim=1)
        else:
            d = torch.ones_like(t)
        st = time.time()
        dist.broadcast(d, src=0, group=group)
        args.comm_time[-1] += time.time() - st
        
        cp.grad.data = d[0]
        # Update server control variate
        scp.data -= d[1] * (len(online_clients) / args.num_workers)
    
    # apply gradient again.
    # Note that when use local SGD, is_global is True for each aggregation.
    optimizer.step(
        apply_lr=False,
        scale=args.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=args.out_momentum,
    )

    # Reassign model to old_model.
    model_server = deepcopy_model(args, model_client)

    # Reassing control variates
    model_client_control = deepcopy_model(args, model_client_control_copy)



    # return deepcopy_model(args, model_server)
    return model_server, model_client_control, model_server_control

def lgt_aggregation(args, model_server, model_client, model_delta, group, online_clients, optimizer, lr, local_steps, lambda_weight=None):
    """Aggregate gradients for local SGD with local gradient tracking.

    Each local model first gets the difference between current model and
    previous synchronized model, and then all-reduce these difference by SUM.
    """
    num_online_clients = len(online_clients) if 0 in online_clients else len(online_clients) + 1
    if (0 not in online_clients) and (args.graph.rank == 0):
        rank_weight = 0
    else:
        if lambda_weight is None:
            rank_weight =  1.0 / num_online_clients
        else:
            #TODO: This is experimental. Test it.
            rank_weight = lambda_weight * args.graph.n_nodes / num_online_clients
    
    for server_param, client_param, delta_param in zip(model_server.parameters(), model_client.parameters(), model_delta.parameters()):
        # get model difference.
        client_param.grad.data = (server_param.data - client_param.data) * rank_weight
        if args.quantized:
            grad_q, q_info = quantize_tensor(client_param.grad.data, adaptive=True)
            gather_list_tensor = [torch.ones_like(client_param.grad.data, dtype=torch.uint8) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
            gather_list_info   = [torch.ones(3) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
            
            st = time.time()
            dist.gather(q_info, gather_list=gather_list_info, dst=0, group=group)
            dist.gather(grad_q, gather_list=gather_list_tensor, dst=0, group=group)
            if args.rounds_comm ==1 and args.graph.rank == 0:
                print("Size of the communication is {}".format(size_tensor(grad_q)))
            delay_time = size_tensor(grad_q) * args.comm_delay_coef
            delay_time += (delay_time/10.0) * np.random.randn()
            time.sleep(delay_time)
            args.comm_time[-1] += time.time() - st
            if args.graph.rank == 0:
                gather_list_tensor = gather_list_tensor if 0 in online_clients else gather_list_tensor[1:]
                gather_list_info = gather_list_info if 0 in online_clients else gather_list_info[1:]
                gather_list_deq = [dequantize_tensor(t,i) for t,i in zip(gather_list_tensor,gather_list_info)]
                d = torch.sum(torch.stack(gather_list_deq,1), dim=1)

                d, avg_info = quantize_tensor(d,adaptive=True)
            else:
                d = torch.ones_like(client_param.grad.data, dtype=torch.uint8)
                avg_info = torch.ones(3)
            
            st = time.time()
            dist.broadcast(avg_info, src=0, group=group)
            dist.broadcast(d, src=0, group=group)
            args.comm_time[-1] += time.time() - st
            client_param.grad.data = dequantize_tensor(d,avg_info)

        else:
            # all reduce to get the average of updates.
            # dist.all_reduce(client_param.grad.data,  op=dist.ReduceOp.SUM, group=group)
            gather_list = [torch.ones_like(client_param.grad.data) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
            st = time.time()
            dist.gather(client_param.grad.data, gather_list=gather_list, dst=0, group=group)
            if args.rounds_comm ==1 and args.graph.rank == 0:
                print("Size of the communication is {}".format(size_tensor(client_param.grad.data)))
            delay_time = size_tensor(client_param.grad.data) * args.comm_delay_coef
            delay_time += (delay_time/10.0) * np.random.randn()
            time.sleep(delay_time)
            args.comm_time[-1] += time.time() - st

            if args.graph.rank == 0:
                gather_list = gather_list if 0 in online_clients else gather_list[1:]
                d = torch.sum(torch.stack(gather_list,1), dim=1)
            else:
                d = torch.ones_like(client_param.grad.data)
            st = time.time()
            dist.broadcast(d, src=0, group=group)
            args.comm_time[-1] += time.time() - st
            client_param.grad.data = d

        # Update the variance reduction control parameter
        if (0 in online_clients) or (args.graph.rank != 0):
            delta_param.data += (server_param.data - client_param.grad.data - client_param.data)/(lr*local_steps)
        # recover to old model.
        client_param.data = server_param.data

    # apply gradient again.
    # Note that when use local SGD, is_global is True for each aggregation.
    optimizer.step(
        apply_lr=False,
        scale=args.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=args.out_momentum,
    )

    # reassign model to old_model.
    model_server = deepcopy_model(args, model_client)
    return model_server, model_delta


def quantize_tensor(x, num_bits=8, adaptive=False, info=None):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    if adaptive:
        min_val, max_val, mean_val = x.min(), x.max(), x.mean()

        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0.0:
            scale=0.001

        initial_zero_point = qmin - (min_val - mean_val) / scale

        zero_point = 0
        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point
        zero_point = int(zero_point)
    else:
        if info is not None:
            scale=info[0]
            zero_point=info[1]
            mean_val=info[2]
        else:
            scale=SCALE_QUANTIZE
            zero_point=ZERO_POINT_QUANTIZE
            mean_val=0.0
    
    q_x = zero_point + (x - mean_val) / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    return q_x, torch.tensor([scale, zero_point, mean_val])


def dequantize_tensor(q_x, info=None):
    if info is None:
        return SCALE_QUANTIZE * (q_x.float() - ZERO_POINT_QUANTIZE)
    else:
        return info[0] * (q_x.float() - info[1]) + info[2]

def size_tensor(x):
    return x.element_size() * x.nelement() / 1e6