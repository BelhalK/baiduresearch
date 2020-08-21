# -*- coding: utf-8 -*-
import numpy as np
import torch
from copy import deepcopy


def get_current_epoch(args):
    if args.growing_batch_size:
        args.epoch_ = args.local_data_seen / args.num_samples_per_epoch
    else:
        args.epoch_ = args.local_index / args.num_batches_train_per_device_per_epoch
    args.epoch = int(args.epoch_)


def get_current_local_step(args):
    """design a specific local step adjustment schme based on lr_decay_by_epoch
    """
    try:
        return args.local_steps[args.epoch]
    except:
        return args.local_steps[-1]


def is_stop(args):
    if args.stop_criteria == 'epoch':
        return args.epoch >= args.num_epochs
    elif args.stop_criteria == 'iteration':
        return args.local_index >= args.num_iterations_per_worker

def update_client_epoch(args):
    args.client_epoch_total += args.local_index / args.num_batches_train_per_device_per_epoch
    return

def zero_copy(model):
    tmp_model = deepcopy(model)
    for tp in tmp_model.parameters():
        tp.data = torch.zeros_like(tp.data)
    return tmp_model