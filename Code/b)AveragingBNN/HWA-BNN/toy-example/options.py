#!/usr/bin/env python

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam', 'hwa'], help='optimization algorithm')
    parser.add_argument('--epochs', type=int, default=1500, help="rounds of training")
    parser.add_argument('--nb_points', type=int, default=300, help="rounds of training")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    #HWA
    parser.add_argument('--start_avg', type=int, default=10, help="Start Averaging in HWA")
    parser.add_argument('--avg_period', type=int, default=10, help="Averaging Period in HWA")

    args = parser.parse_args()
    return args
