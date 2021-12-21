#!/usr/bin/env python

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--dim', type=int, default=2, help="nb of dimensions")
    parser.add_argument('--cuda', type=str, default='False',help="GPU")
    parser.add_argument('--paths', type=str, default='mnist', help="paths to images or ckpt")
    args = parser.parse_args()
    return args
