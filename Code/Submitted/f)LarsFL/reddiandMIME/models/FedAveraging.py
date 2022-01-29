#!/usr/bin/env python

import copy
import torch
from torch import nn
import collections

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = w_avg[k]/len(w)
    return w_avg

def FedAvg_num_key(w):
    w_avg = collections.OrderedDict()
    for i,k in enumerate(w[0].keys()):
        w_avg[i] = torch.zeros(w[0][k].shape)
        
    for j,k in enumerate(w[0].keys()):
        for i in range(len(w)):
            w_avg[j] += w[i][k]
        w_avg[j] = w_avg[j]/len(w)
    return w_avg

def FedMax(x,y):
    m = copy.deepcopy(x)
    for ke in m.keys():
        m[ke] = torch.max(x[ke],y[ke])
    
    return m

def FedMinus(x,y):
    m = copy.deepcopy(x)
    for ke,ke2 in zip(m.keys(),y.keys()):
        m[ke] = x[ke]-y[ke2]
    
    return m