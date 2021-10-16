#!/usr/bin/env python

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="Total number of clients: K")
    parser.add_argument('--frac', type=float, default=1, help="Fraction of Clients used in each round: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local mini-batch size (SGD): B")
    parser.add_argument('--bs', type=int, default=500, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--wdecay', type=float, default=0, help="amsrgad weight decay")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument('--beta2', default=0.999, type=float, help='betar2 EMA var scale for ADAM')
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    parser.add_argument('--method', default='adapt', type=str, choices=['topk','unif','tong','sign','full'], help='optimization algorithm')
    parser.add_argument('--order', default=0.5, type=float, help='order factor xi')
    parser.add_argument('--u_withQ', default=1, type=int, help='whether counting factor with tilde')
    parser.add_argument('--error', default=0, type=int, help='error compensation')
    parser.add_argument('--error_gamma', default=1, type=float, help='error accumulation factor')
    
    parser.add_argument('--sparsity', type=float, default=0.05, help="gradient sparsity")
#    parser.add_argument('--delay', type=float, default=0, help="whether delay gradient")
    parser.add_argument('--tau', type=int, default=0, help="number of delays")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--customarg', type=int, default=1, help="")
    args = parser.parse_args()
    return args
