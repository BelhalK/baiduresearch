#!/usr/bin/env python
# python3 main2.py --dataset mnist --num_channels 1 --model cnn --epochs 2 --gpu -1 --optim localams --num_users 20 --frac 0.5 --local_ep 1 --glob_lr 0.1 --lr 0.0001 --customarg 0
# python3 main2.py --dataset mnist --num_channels 1 --model cnn --epochs 2 --gpu -1 --optim sgd --LAMB 0 --num_users 20 --frac 0.5 --local_ep 1 --glob_lr 0.1 --lr 0.0001 --customarg 0

#runfile('C:/Users/bdfzl/Desktop\PhD/larsFL with Belhal/python code/main2.py', wdir='C:/Users/bdfzl/Desktop\PhD/larsFL with Belhal/python code',args='--dataset cifar --num_channels 1 --model cnn --epochs 20 --gpu -1 --optim sgd --num_users 60 --frac 0.5 --local_ep 1 --glob_lr 1 --lr 0.1 --customarg 1 --LAMB 0 --lambda0 0.01')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import collections
from tensorflow.keras.datasets import fashion_mnist

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, fmnist_noniid, cifar_noniid2
from utils.options import args_parser
from models.LocUpdate2 import LocalUpdate, LocalUpdateAMS
from models.Models import MLP, CNNMnist, CNNCifar, ResNet18, ResNet9
from models.FedAveraging import FedAvg, FedMax, FedMinus, FedAvg_num_key
from models.Testing import test_fct
from models.global_model import GlobalAMSGrad
import torchvision

from logger import Logger, savefig
import pdb 
import time
#from tensorflow.examples.tutorials.mnist import input_data
#import mnist_reader
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

# import sys
# sys.setrecursionlimit(100000)

#mnist mlp: iid1  0.001   LAMB    iid0: 0.0001  LAMB 0.01

if __name__ == '__main__':
    
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # load dataset and split data (iid or not) to users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        aa=dataset_train.train_data; ll=dataset_train.train_labels
        print(aa.shape)
        dataset_train = TensorDataset(dataset_train.train_data.unsqueeze(1).type(torch.FloatTensor),dataset_train.train_labels)
        dataset_test = TensorDataset(dataset_test.test_data.unsqueeze(1).type(torch.FloatTensor),dataset_test.test_labels)
        # sample users
        if args.customarg==1:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print('non-iid data!')
            dict_users = fmnistcconiid(ll, args.num_users)
    elif args.dataset == 'cifar':#num_channels == 3
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        aa=dataset_train.data; ll=dataset_train.targets
        print(aa.shape)
        dataset_train = TensorDataset(Tensor(dataset_train.data).permute(0,3,1,2).type(torch.FloatTensor),Tensor(dataset_train.targets).type(torch.LongTensor))
        dataset_test = TensorDataset(Tensor(dataset_test.data).permute(0,3,1,2).type(torch.FloatTensor),Tensor(dataset_test.targets).type(torch.LongTensor))
        # sample users
        if args.customarg==1:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            print('non-iid data!')
            dict_users = cifar_noniid2(ll, args.num_users)
        #if args.customarg==1:
        #    dict_users = cifar_iid(dataset_train, args.num_users)
        #else:
        #    dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fmnist':
        (X_train, train_labels), (X_test, test_labels) = fashion_mnist.load_data()
        print(X_train.shape)
        X_train = np.expand_dims(np.reshape(X_train,(-1,28,28)),1)/255
        X_test = np.expand_dims(np.reshape(X_test,(-1,28,28)),1)/255
        dataset_train = TensorDataset(Tensor(X_train), Tensor(train_labels).type(torch.LongTensor))
        dataset_test = TensorDataset(Tensor(X_test), Tensor(test_labels).type(torch.LongTensor))
        if args.customarg==1:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print('non-iid data!')
            dict_users = fmnist_noniid(train_labels, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape


    # build global model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fmnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model =='resnet':
        net_glob = ResNet9().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    

    print(net_glob)
    net_glob.train()

    
    # Prepare Logger file
    logname = args.model
    dataset = args.dataset
    title = '{}-{}'.format(dataset, logname)
    checkpoint_dir = 'checkpoints/new/checkpoints_{}'.format(dataset)
    if args.LAMB:
        logger = Logger('{}/log{}_opt{}_lambda{}_lr{}_ep{}_clients{}_frac{}_iid{}_seed{}.txt'.format(checkpoint_dir, logname,  args.optim, args.LAMB, args.lambda0, args.lr, args.local_ep,args.num_users, args.frac,args.customarg,args.seed), title=title)
    else:
        logger = Logger('{}/log{}_opt{}_globlr{}_loclr{}_ep{}_clients{}_frac{}_iid{}_seed{}.txt'.format(checkpoint_dir, logname,  args.optim, args.glob_lr, args.lr, args.local_ep,args.num_users, args.frac,args.customarg,args.seed), title=title)        
    logger.set_names(['Learning Rate', 'Avg. Loss','Train Loss','Train Acc.','Test Loss','Test Acc.', 'Time'])

    # training
    loss_train_avg = []
    
    start_time = time.time()
    
    n = max(int(args.frac * args.num_users), 1)
    
    
    glob = GlobalAMSGrad(net_glob.parameters(), lr=args.glob_lr,device=args.device, 
                                            betas=(args.momentum, args.beta2), 
                                            eps=1e-8, weight_decay=args.wdecay, amsgrad=False)
    
    for iter in range(args.epochs):
        w_diff_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), n, replace=False)
        
        glob_params = net_glob.state_dict()
        # glob_params = collections.OrderedDict()
        # for i, parameter in enumerate(net_glob.parameters()):
        #     glob_params[i] = parameter.data
        
        #local updates/training
        for idx in idxs_users:
            print('Starting for worker nb {:3d}'.format(idx))
            if args.optim == 'reddi':
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            loss_locals.append(copy.deepcopy(loss))
            w_diff_locals.append(copy.deepcopy(FedMinus(w,glob_params)))

        eff_grad = FedAvg_num_key(w_diff_locals)
        print(len(eff_grad))
        glob.step(eff_grad)
        
        #Global parameters update
        # glob_params = FedAvg(w_diff_locals)
        
        # # copy global parameters to net_glob (for broadcast to all users)
        # net_glob.load_state_dict(glob_params)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train_avg.append(loss_avg)

        # testing
        net_glob.eval()
        acc_train, loss_train = test_fct(net_glob, dataset_train, args)
        acc_test, loss_test = test_fct(net_glob, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))

        elapsed_time = time.time() - start_time
        logger.append([args.lr, loss_avg,loss_train, acc_train, loss_test, acc_test, elapsed_time])

    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train_avg)), loss_train_avg)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))


#for lr in lr_list:
#    for ep in ep_list:
#        for lamb in lambda_list:
#            runfile('C:/Users/bdfzl/Desktop\PhD/larsFL with Belhal/python code/main2.py', wdir='C:/Users/bdfzl/Desktop\PhD/larsFL with Belhal/python code',args='--dataset cifar --num_channels 1 --model cnn --epochs 20 --gpu -1 --optim sgd --num_users 60 --frac 0.5 --local_ep '+ ep +' --glob_lr 1 --lr '+ lr +' --customarg 1 --LAMB 1 --lambda0 '+ lamb)
 
#for lr in lr_list:
#    for ep in ep_list:
#            runfile('C:/Users/bdfzl/Desktop\PhD/larsFL with Belhal/python code/main2.py', wdir='C:/Users/bdfzl/Desktop\PhD/larsFL with Belhal/python code',args='--dataset cifar --num_channels 1 --model cnn --epochs 20 --gpu -1 --optim sgd --num_users 60 --frac 0.5 --local_ep '+ ep +' --glob_lr 1 --lr '+ lr +' --customarg 1 --LAMB 0 --lambda0 0')
  
