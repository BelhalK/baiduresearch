#!/usr/bin/env bash
# 1 activation of convergence diagnostic 
ipython main.py -- --dataset cifar10 --epochs 150 --schedule diagnostic --lr 0.001 --momentum 0 --weight_decay 0 --optimizer signSGD --gpu 5 4 --logname cifar.resnet.sign.diag.072319 --burnin 20 --num_reduce 1
