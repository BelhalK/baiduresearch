#!/usr/bin/env python
# python3 main.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 2 --gpu -1 --optim sketchedsgd --num_users 10 --local_ep 2

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
from models.LocUpdate import LocalUpdate, LocalUpdateSketch
from models.Models import MLP, CNNMnist, CNNCifar
from models.FedAveraging import FedAvg, SkeAvg
from models.Testing import test_fct
from models.sketchedsgd_opt import SketchedSGD, SketchedSum, SketchedModel

from logger import Logger, savefig
import pdb 
import time

import models.sketch_operations as sko

# import sys
# sys.setrecursionlimit(100000)

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split data (iid or not) to users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
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
    
    #if  sketch (fedsketch or sketchedsgd) then SketchedModel
    if args.optim == 'sketchedsgd':
        net_glob = SketchedModel(net_glob)

    # print(net_glob)
    net_glob.train()

    torch.manual_seed(123)
    # Prepare Logger file
    logname = args.model
    dataset = args.dataset
    title = '{}-{}'.format(dataset, logname)
    checkpoint_dir = 'checkpoints/checkpoints_{}'.format(dataset)
    logger = Logger('{}/log{}_opt{}_sketch{}_lr{}_globlr{}_bs{}_clients{}_frac{}.topk{}_t{}_k{}_localep{}.txt'.format(checkpoint_dir, logname,  args.optim, args.sketching, args.lr,args.glob_lr, args.local_bs,args.num_users, args.frac, args.topk, args.arraysize, args.binsize, args.local_ep), title=title)
    logger.set_names(['Learning Rate', 'Avg. Loss','Train Loss','Train Acc.','Test Loss','Test Acc.', 'Time'])

    # global parameters
    glob_params = net_glob.state_dict()
    # training
    loss_train_avg = []
    
    start_time = time.time()


    #Sketching arguments (size)
    t = args.arraysize 
    k = args.binsize

    #store shape of tensor per layer (for reshaping the numpy arrays)
    d = collections.OrderedDict()
    
    ####hashes and signs common to all users
    hashfct = collections.OrderedDict()
    sign = collections.OrderedDict()
    for i, ke in enumerate(glob_params.keys()):
        d[ke] = glob_params[ke].shape
        dim = glob_params[ke].numel() #total dimension per layer
        hashfct[ke] = np.random.randint(k,size=(t,dim))
        sign[ke] = np.sign(np.random.rand(t,dim)-0.5)
    
    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        ##local updates/training
        for idx in idxs_users:
            # print('Starting for worker nb {:3d}'.format(idx))
            if args.optim == 'sketchedsgd': #topk done already
                local = LocalUpdateSketch(args=args, dataset=dataset_train, idxs=dict_users[idx])
            else: #local sgd or fedsketch
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            loss_locals.append(copy.deepcopy(loss))
            w_locals.append(copy.deepcopy(w))

        ### local weights sketching before doing Averaging on the parameter server
        if args.optim == 'fedsketch':
            sketch_locals = []
            for i, weights in enumerate(w_locals): #sketching per local weight
                sketched_w = collections.OrderedDict()
                #sketch the weight per layer (arg needs to be a vector)
                # workerHash = hashes[i]
                # workerSign = signs[i]
                for ke in weights.keys(): #sketching per layer (for scaling)
                    init_weights = glob_params[ke].cpu().detach().numpy().ravel()
                    curr_weights = weights[ke].cpu().detach().numpy().ravel()
                    vec =  init_weights - curr_weights
                    # sketch = sko.generate_sketch(vec, t, k , workerHash[ke], workerSign[ke])
                    sketch = sko.generate_sketch(vec, t, k , hashfct[ke], sign[ke])
                    sketched_w[ke] = sketch
                sketch_locals.append(sketched_w)
            
            # Averaging all the workers sketches
            sketch_avg = SkeAvg(sketch_locals)
            
            # apply the correct method (privix, topk or heaprix) to the aggregated sketch
            if args.sketching == 'privix':
                glob_sketch = copy.deepcopy(sketch_avg)
                for ke in glob_sketch.keys():
                    dim = glob_params[ke].numel() #total dimension per layer
                    glob_sketch[ke]=sko.PRIVIX(sketch_avg[ke],dim,hashfct[ke], sign[ke])
            elif args.sketching == 'topk':
                glob_sketch = copy.deepcopy(sketch_avg)
                frac_topk = args.topk
                for ke in glob_sketch.keys():
                    dim = glob_params[ke].numel() 
                    glob_sketch[ke]=sko.HEAVYMIX(sketch_avg[ke], frac_topk, dim,hashfct[ke], sign[ke] )
            elif args.sketching == 'heaprix':
                tmp_glob_sketch = copy.deepcopy(sketch_avg)
                frac_topk = args.topk
                # HEAVYMIX in line 14
                for ke in tmp_glob_sketch.keys():
                    dim = glob_params[ke].numel() 
                    # Compute global TOP K
                    tmp_glob_sketch[ke]=sko.HEAVYMIX(sketch_avg[ke], frac_topk, dim,hashfct[ke], sign[ke] )
                
                heavymix_locals = []
                for i, weights in enumerate(w_locals): #sketching per local weight
                    sketched_heav_locals = collections.OrderedDict()
                    for ke in weights.keys():
                        # Compute Sketching in line 14 (for each device j)
                        sketched_heav_locals[ke]= sko.generate_sketch(tmp_glob_sketch[ke], t, k , hashfct[ke], sign[ke])
                    heavymix_locals.append(sketched_heav_locals)
                
                # line 15 of Alg 4 (averaging of the sketches of HEAVYMIX)
                heavymix_avg = SkeAvg(heavymix_locals)
                
                # Update global drift term before Descent
                glob_sketch = copy.deepcopy(sketch_avg)
                for ke in glob_sketch.keys():
                    dim = glob_params[ke].numel()
                    #line 4 of algorithm 5
                    tmp_heavy = sko.HEAVYMIX(sketch_avg[ke], frac_topk, dim,hashfct[ke], sign[ke] )
                    tmp_priv = sko.PRIVIX(sketch_avg[ke] - heavymix_avg[ke],dim,hashfct[ke], sign[ke])
                    glob_sketch[ke] = tmp_heavy + tmp_priv
    
        #Global parameters update
        if args.optim == 'fedsketch':
            for ke in glob_params.keys():
                #global gradient step using Averaged Sketch (with privix etc.. applied)
                tmp_glob_sketch = torch.from_numpy(np.reshape(glob_sketch[ke], d[ke]))
                glob_params[ke] -= args.glob_lr*tmp_glob_sketch
        else:
            # update global parameters using averaging (FedAvg)
            # argument of FedAvg must be an list (over users) of orderedDict 
            glob_params = FedAvg(w_locals)
        
        # copy global parameters to net_glob (for broadcast to all users)
        net_glob.load_state_dict(glob_params)

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

    