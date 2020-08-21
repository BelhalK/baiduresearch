# -*- coding: utf-8 -*-
import time

import torch
import numpy as np

from pcode.tracking.logging import log
from pcode.components.datasets.partition import DataPartitioner, FederatedPartitioner
from pcode.components.datasets.prepare_data import get_dataset


def _load_data_batch(args, _input, _target):
    if 'least_square' in args.arch:
        _input = _input.float()
        _target = _target.unsqueeze_(1).float()
    else:
        if 'epsilon' in args.data or 'url' in args.data or 'rcv1' in args.data or 'higgs' in args.data:
            _input, _target = _input.float(), _target.long()

    if args.graph.on_cuda:
        _input, _target = _input.cuda(), _target.cuda()
    return _input, _target


def load_data_batch(args, _input, _target, tracker):
    """Load a mini-batch and record the loading time."""
    # get variables.
    start_data_time = time.time()

    _input, _target = _load_data_batch(args, _input, _target)

    # measure the data loading time
    end_data_time = time.time()
    tracker['data_time'].update(end_data_time - start_data_time)
    tracker['end_data_time'] = end_data_time
    return _input, _target


def define_dataset(args, shuffle):
    log('create {} dataset for rank {}'.format(args.data, args.graph.rank), args.debug)
    train_loader = partition_dataset(args, shuffle, dataset_type='train')
    if args.fed_personal:
        train_loader, val_loader = train_loader
    test_loader = partition_dataset(args, shuffle, dataset_type='test')

    get_data_stat(args, train_loader, test_loader)
    if args.fed_personal:
        return train_loader, test_loader, val_loader
    else:
        return train_loader, test_loader

def partitioner(args, dataset, shuffle, world_size, partition_type='normal'):
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    if partition_type == 'normal':
        partition = DataPartitioner(args, dataset, shuffle, partition_sizes)
    elif partition_type == 'noniid':
        partition = FederatedPartitioner(args,dataset, shuffle)
    return partition.use(args.graph.rank)


def partition_dataset(args, shuffle, dataset_type='train'):
    """ Given a dataset, partition it. """
    dataset = get_dataset(args, args.data, args.data_dir, split=dataset_type)
    batch_size = args.batch_size
    world_size = args.graph.n_nodes

    # partition data.
    if args.partition_data and args.iid_data:
        data_to_load = partitioner(args, dataset, shuffle, world_size)
        log('partitioned data and use subdata.', args.debug)
    elif not args.iid_data and dataset_type == 'train':
        data_to_load = partitioner(args, dataset, shuffle,world_size, partition_type='noniid')
        log('Make federated data partitions and use the subdata.', args.debug)
    else:
        data_to_load = dataset
        log('used whole data.', args.debug)

    if dataset_type == 'train':
        args.train_dataset_size = len(dataset)
        log('  We have {} samples for {}, \
            load {} data for process (rank {}).'.format(
            len(dataset), dataset_type, len(data_to_load), args.graph.rank), args.debug)
    else:
        args.val_dataset_size = len(dataset)
        log('  We have {} samples for {}, \
            load {} val data for process (rank {}).'.format(
            len(dataset), dataset_type, len(data_to_load), args.graph.rank), args.debug)

    if dataset_type == 'train':
        if args.stop_criteria == 'epoch':
            args.num_iterations = int(len(data_to_load) * args.num_epochs / batch_size)
        else:
            args.num_epochs = int(args.num_iterations * batch_size / len(data_to_load))
        args.total_data_size = len(data_to_load) * args.num_epochs
        args.num_samples_per_epoch = len(data_to_load)
    # use Dataloader.
    data_type_label = (dataset_type == 'train')
    if args.fed_personal and data_type_label:
        val_size = int(0.2*len(data_to_load))
        data_to_load_train, data_to_load_val = torch.utils.data.random_split(data_to_load,[len(data_to_load) - val_size, val_size])
        data_loader_train = torch.utils.data.DataLoader(
                                data_to_load_train, batch_size=batch_size,
                                shuffle=data_type_label,
                                num_workers=1, pin_memory=args.pin_memory,
                                drop_last=False)
        data_loader_val = torch.utils.data.DataLoader(
                                data_to_load_val, batch_size=batch_size,
                                shuffle=data_type_label,
                                num_workers=1, pin_memory=args.pin_memory,
                                drop_last=False)
        data_loader = [data_loader_train, data_loader_val]
        log('we have {} batches for {} for rank {}.'.format(
            len(data_loader[0]), 'train', args.graph.rank), args.debug)
        log('we have {} batches for {} for rank {}.'.format(
            len(data_loader[1]), 'val', args.graph.rank), args.debug)
    else:
        data_loader = torch.utils.data.DataLoader(
            data_to_load, batch_size=batch_size,
            shuffle=data_type_label,
            num_workers=1, pin_memory=args.pin_memory,
            drop_last=False)
        log('we have {} batches for {} for rank {}.'.format(
            len(data_loader), dataset_type, args.graph.rank), args.debug)
    return data_loader


def get_data_stat(args, train_loader, test_loader):
    args.num_batches_train_per_device_per_epoch = len(train_loader)
    args.num_whole_train_batches_per_worker = \
        args.num_batches_train_per_device_per_epoch * args.num_epochs
    args.num_warmup_train_batches_per_worker = \
        args.num_batches_train_per_device_per_epoch * args.lr_warmup_epochs
    args.num_iterations_per_worker = args.num_iterations #// args.graph.n_nodes

    # get the data statictics (on behalf of each worker) for val.
    args.num_batches_val_per_device_per_epoch = len(test_loader)

    # define some parameters for training.
    log('we have {} epochs, \
        {} mini-batches per device for training. \
        {} mini-batches per device for test. \
        The batch size: {}.'.format(
            args.num_epochs,
            args.num_batches_train_per_device_per_epoch,
            args.num_batches_val_per_device_per_epoch,
            args.batch_size), args.debug)
