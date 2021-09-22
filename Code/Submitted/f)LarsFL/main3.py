#!/usr/bin/env python
# python3 main.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 2 --gpu -1 --optim sgd --num_users 10 --local_ep 1
# python3 main.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 2 --gpu -1 --optim localams --num_users 10 --local_ep 1

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import collections

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.LocUpdate3 import LocalUpdate, LocalUpdateAMS
from models.Models import MLP, CNNMnist, CNNCifar
from models.FedAveraging import FedAvg, FedMax
from models.Testing import test_fct
from models.localams3 import AMSGrad_LocalOPT

from logger import Logger, savefig
import pdb 
import time


# import sys
# sys.setrecursionlimit(100000)

if __name__ == '__main__':
    
    torch.manual_seed(0)
    
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # load dataset and split data (iid or not) to users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.customarg==1:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print('non-iid data!')
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':#num_channels == 3
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build global model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
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
    checkpoint_dir = 'checkpoints/checkpoints_{}'.format(dataset)
    logger = Logger('{}/log{}_opt{}_lr{}_ep{}_clients{}_frac{}_iid{}.txt'.format(checkpoint_dir, logname,  args.optim, args.lr, args.local_ep,args.num_users, args.frac,args.customarg), title=title)
    logger.set_names(['Learning Rate', 'Avg. Loss','Train Loss','Train Acc.','Test Loss','Test Acc.', 'Time'])

    # global parameters
    glob_params = net_glob.state_dict()
    # training
    loss_train_avg = []
    
    start_time = time.time()

    #store shape of tensor per layer (for reshaping the numpy arrays)
    d = collections.OrderedDict()
    glob_v_hat = collections.OrderedDict()
    each_v = collections.OrderedDict()
    
    for i, ke in enumerate(glob_params.keys()):
        d[i] = glob_params[ke].shape
        glob_v_hat[i] = torch.ones(d[i]) * 1e-8
        each_v[i] = torch.ones(d[i]) * 1e-8
        
    # if args.optim == 'fedsketch' and args.sketching == 'nips19':
    #     errors=[]
    #     for i in range(args.num_users):
    #         gg=collections.OrderedDict()
    #         for j, ke in enumerate(glob_params.keys()):
    #             gg[ke]=np.zeros(d[ke]).ravel()
            
    #         errors.append(gg)
    
    m = max(int(args.frac * args.num_users), 1)
    
    local_v = [copy.deepcopy(each_v)]*args.num_users  #keep v for every local worker
    
    OPTs = [AMSGrad_LocalOPT(net =copy.deepcopy(net_glob).to(args.device), params = copy.deepcopy(net_glob).to(args.device).parameters(), lr=args.lr, betas=(args.momentum, args.beta2), 
                                            eps=1e-8, weight_decay=args.wdecay, amsgrad=True, 
                                            v_hat=glob_v_hat, num_round=0) for i in range(args.num_users)]
    
    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        #local updates/training
        for idx in idxs_users:
            print('Starting for worker nb {:3d}'.format(idx))
            if args.optim == 'localams': #topk done already
                local = LocalUpdateAMS(args=args, dataset=dataset_train, idxs=dict_users[idx])
            else: #local sgd 
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, v = local.train( optimizer=OPTs[idx])
            loss_locals.append(copy.deepcopy(loss))
            w_locals.append(copy.deepcopy(w))
            local_v[idx] = v
            
        average_v = FedAvg([local_v[i] for i in idxs_users])          
        glob_v_hat = FedMax(glob_v_hat, average_v)
        
        #Global parameters update
        glob_params = FedAvg(w_locals)
        
        # copy global parameters to net_glob (for broadcast to all users)
        net_glob.load_state_dict(glob_params)
        
        for i in range(args.num_users):
#            if i in idxs_users:
            OPTs[i].update_params(net_glob.parameters(),glob_v_hat,iter+1)
#            else:
#                OPTs[i].update_params(copy.deepcopy(net_glob).to(args.device).train().parameters(),glob_v_hat,iter)
                
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

    