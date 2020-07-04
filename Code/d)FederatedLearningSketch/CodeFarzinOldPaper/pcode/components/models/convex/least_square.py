# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


__all__ = ['least_square']


class Least_square(torch.nn.Module):

    def __init__(self, dataset):
        super(Least_square, self).__init__()
        self.dataset = dataset

        # get input and output dim.
        self._determine_problem_dims()

        # define layers.
        self.fc = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_classes, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x

    def _determine_problem_dims(self):
        if self.dataset == 'epsilon':
            self.num_features = 2000
            self.num_classes = 1
        elif self.dataset == 'url':
            self.num_features = 3231961
            self.num_classes = 1
        elif self.dataset == 'rcv1':
            self.num_features = 47236
            self.num_classes = 1
        elif self.dataset == 'MSD':
            self.num_features = 90
            self.num_classes = 1
        else:
            raise ValueError('convex methods only support epsilon, url, YearPredictionMSD and rcv1 for the moment')


def least_square(args):
    return Least_square(dataset=args.data)
