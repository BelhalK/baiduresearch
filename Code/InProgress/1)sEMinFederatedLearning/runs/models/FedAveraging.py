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

def SkeAvg(sketch_locals):
    sketch_avg = copy.deepcopy(sketch_locals[0])
    for ke in sketch_avg.keys():
        for i in range(len(sketch_locals)):
            sketch_avg[ke] += sketch_locals[i][ke]
        sketch_avg[ke] = sketch_avg[ke]/len(sketch_locals)
    return sketch_avg

