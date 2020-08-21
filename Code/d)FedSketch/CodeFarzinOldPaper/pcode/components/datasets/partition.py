# -*- coding: utf-8 -*-
import random

import torch
import torch.distributed as dist
import numpy as np


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return self.data[data_idx]


class Partitioner(object):
    def consistent_indices(self, indices, shuffle):
        if self.args.graph.rank == 0 and shuffle:
            random.shuffle(indices)

        # broadcast.
        indices = torch.IntTensor(indices)
        group = dist.new_group(self.args.graph.ranks)
        dist.broadcast(indices, src=0, group=group)
        return list(indices)
    def check_indices(self, indices):
        t_indices = torch.IntTensor(indices)
        group = dist.new_group(self.args.graph.ranks)
        dist.broadcast(indices, src=0, group=group)
        if not torch.equal(t_indices, torch.IntTensor(indices)):
            raise ValueError("Data chuncks in different devices are not the same!")
        return

class DataPartitioner(Partitioner):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, args, data, shuffle, sizes=[0.7, 0.2, 0.1]):
        # prepare info.
        self.args = args
        self.data = data
        self.data_size = len(self.data) 
        self.partitions = []

        # get shuffled/unshuffled data.
        indices = [x for x in range(0, self.data_size)]

        indices = self.consistent_indices(indices, shuffle)

        # partition indices.
        from_index = 0
        for ind, frac in enumerate(sizes):
            to_index = from_index + int(sizes[ind] * self.data_size)
            self.partitions.append(indices[from_index: to_index])
            from_index = to_index

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind])


class FederatedPartitioner(Partitioner):
    """ Partitions a dataset into different chuncks to make data non-iid for federated learning."""
    def __init__(self, args, data, shuffle, sizes=None):
        # prepare info.
        del sizes
        self.args = args
        self.data = data
        self.data_size = len(self.data)
        self.partitions = []

        # If data is synthetic, the chunk of each client is decided beforehand.
        if args.data == 'synthetic':
            self.partitions = [[] for _ in range(args.graph.n_nodes)]
            indices = data.indices
            # self.check_indices(indices)
            indices = np.insert(indices,0,0)
            from_to_indices = list(zip(indices[: -1], indices[1:]))
            for from_index, to_index in from_to_indices:
                self.partitions[args.graph.rank].extend(list(range(from_index, to_index)))

        else:
            self.labels = self.data.train_labels
            self.classes = self.labels.unique()
            if args.unbalanced:
                min_size = int(self.data_size / (len(self.classes) * self.args.graph.n_nodes))
                # max_size = int(self.data_size / (self.args.num_class_per_client * self.args.graph.n_nodes)) * 2 - 100
                slice_sizes = min_size * np.ones((self.args.num_class_per_client, self.args.graph.n_nodes ), dtype=int)
                for i in range(self.args.num_class_per_client):
                    total_remainder = int(self.data_size / self.args.num_class_per_client) - min_size * self.args.graph.n_nodes
                    ind = np.sort(np.random.choice(np.arange(0,total_remainder), self.args.graph.n_nodes - 1, replace=False))
                    ind = np.insert(ind,0,0)
                    ind = np.insert(ind, len(ind),total_remainder)
                    class_sizes = ind[1:] - ind[:-1]
                    slice_sizes[i,:] += class_sizes
            else: 
                slice_size = int(self.data_size / (self.args.graph.n_nodes * self.args.num_class_per_client))
                slice_sizes = np.zeros((self.args.num_class_per_client,self.args.graph.n_nodes), dtype=int)
                slice_sizes += slice_size
    
            # get shuffled/unshuffled data.
            indices = self.sort_labels()

            indices = self.consistent_indices(indices, shuffle=False)

            # partition indices.
            from_index = 0
            for n_class in range(self.args.num_class_per_client):
                for client in range(self.args.graph.n_nodes):
                    to_index = from_index + slice_sizes[n_class, client]
                    if n_class == 0:
                        self.partitions.append(indices[from_index: to_index])
                    else:
                        self.partitions[client].extend(indices[from_index: to_index])
                    from_index = to_index
        # TODO: add shuffling each clients data

    def sort_labels(self):
        label_array = self.labels.numpy()
        class_array = self.classes.numpy()
        sorted_ind = np.concatenate([np.squeeze(np.argwhere(label_array == c)) for c in class_array], axis=0)
        return list(sorted_ind)

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind])