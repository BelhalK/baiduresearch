#!/usr/bin/env bash
# normal run to see how good signSGD does
ipython main.py -- --dataset cifar10 --epochs 150 --schedule 80 120 --lr 0.001 --momentum 0 --weight_decay 0 --optimizer signSGD --gpu 7 6 --logname cifar.resnet.sign1.072319
