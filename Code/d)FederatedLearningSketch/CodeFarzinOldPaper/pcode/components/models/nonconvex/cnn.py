# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F


__all__ = ['cnn']

class CNN(nn.Module):
    def __init__(self,dataset):
        super(CNN, self).__init__()
        self.dataset = dataset
        self.num_classes = self._decide_num_classes()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(4*4*50, 512)
        self.fc2 = nn.Linear(512, 10)



    def _decide_num_classes(self):
        if self.dataset in ['cifar10', 'mnist']:
            return 10
        elif self.dataset == 'cifar100':
            return 100

    def _decide_input_feature_size(self):
        if 'mnist' in self.dataset:
            return 28 * 28
        else:
            raise NotImplementedError


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def cnn(args):
    return CNN(args.data)