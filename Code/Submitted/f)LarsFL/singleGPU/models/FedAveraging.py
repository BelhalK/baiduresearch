#!/usr/bin/env python

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = w_avg[k]/len(w)
    return w_avg

def FedMax(x,y):
    m = copy.deepcopy(x)
    for ke in m.keys():
        m[ke] = torch.max(x[ke],y[ke])
    
    return m
