import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import collections

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.LocUpdate2 import LocalUpdateAMS
from models.localams2 import LocalAMSGrad

from models.Models import MLP, CNNMnist, CNNCifar, LeNet, LeNet_cifar, Logistic, ResNet18, ResNet9, LSTMNet, ResNet50
from models.ModelsTiny import resnet18

from models.FedAveraging import FedAvg
from models.Testing import test_fct
import utils.sparsification as sparse

from logger import Logger, savefig
import pdb 
import time

import models.sketch_operations as sko
# from tensorflow.python.keras.datasets import imdb
# from tensorflow.python.keras.preprocessing import sequence
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

# import sys
# sys.setrecursionlimit(100000)
# 1 worker: bs 200   more: bs 500
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
      m.eval()
    
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
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    
        # Normalize the test set same as training set without augmentation
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform_test)
        if args.customarg==1:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'imdb':
        args.num_classes = 2
        top_words=1000
        max_review_len=400
        embedding_length=32
        num_cells=32
        (X_train,label_train),(X_test,label_test) = imdb.load_data(num_words=top_words)
        X_train=sequence.pad_sequences(X_train,maxlen=max_review_len,padding='post')
        X_test=sequence.pad_sequences(X_test,maxlen=max_review_len,padding='post')
        dataset_train = TensorDataset(Tensor(X_train), Tensor(label_train).type(torch.LongTensor))
        dataset_test = TensorDataset(Tensor(X_test), Tensor(label_test).type(torch.LongTensor))
    elif args.dataset == 'tinyimagenet':
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        trans_tiny_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
        dataset_train = datasets.ImageFolder('data/tiny-imagenet-200/train', transform=trans_tiny_train)
        # dataset_train= DataLoader(train, batch_size=args.bs)

        trans_tiny_test = transforms.Compose([transforms.ToTensor(), norm])
        dataset_test = datasets.ImageFolder('data/tiny-imagenet-200/test', transform=trans_tiny_test)
        # dataset_test= DataLoader(test, batch_size=args.bs)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build global model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'lenet':
        net_glob = LeNet(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model =='resnet9':
        net_glob = ResNet9().to(args.device)
    elif args.model =='resnet18':
        # net_glob = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # net_glob = net_glob.to(args.device)
        # net_glob = ResNet18().to(args.device)
        net_glob = resnet18().to(args.device)
    elif args.model == 'resnet50':
        net_glob = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        net_glob = net_glob.to(args.device)
        # net_glob = ResNet50().to(args.device)
    elif args.model == 'resnet101':
        net_glob = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        net_glob = net_glob.to(args.device)
        # net_glob = ResNet101().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'logistic':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = Logistic(dim_in=len_in, dim_out=args.num_classes).to(args.device)
    elif args.model == 'lstm':
        net_glob = LSTMNet(embedding_dim=embedding_length, hidden_dim=num_cells, vocab_size=top_words, tagset_size=2).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    print(net_glob)
    net_glob.apply(set_bn_eval)
    net_glob.train()
    net_glob.apply(set_bn_eval)
    stale_net_glob = copy.deepcopy(net_glob)
    stale_net_glob.train()

    
    # Prepare Logger file
    logname = args.model
    dataset = args.dataset
    title = '{}-{}'.format(dataset, logname)
    checkpoint_dir = 'checkpoints/new/checkpoints_{}'.format(dataset)
    if args.method == 'full':
        logger = Logger('{}/log{}_method{}_lr{}_clients{}_frac{}_tau{}_epoch{}.txt'.format(checkpoint_dir, logname,  args.method, args.lr,args.num_users, args.frac, args.tau, args.epochs), title=title)
    elif args.method != 'sign':
        logger = Logger('{}/log{}_method{}_lr{}_clients{}_frac{}_sparsity{}_tau{}_error{}_epoch{}.txt'.format(checkpoint_dir, logname,  args.method, args.lr, args.num_users, args.frac,args.sparsity, args.tau, args.error, args.epochs), title=title)
    else:
        logger = Logger('{}/log{}_method{}_lr{}_clients{}_frac{}_tau{}_error{}_epoch{}.txt'.format(checkpoint_dir, logname,  args.method, args.lr, args.num_users, args.frac, args.tau, args.error, args.epochs), title=title)
    logger.set_names(['Learning Rate', 'Train Loss','Train Acc.','Test Loss','Test Acc.', 'Time'])

    # global parameters
    # glob_params = net_glob.state_dict()
    glob_params = collections.OrderedDict()
    for i,p in enumerate(net_glob.parameters()):
        glob_params[i] = p
    
    # training
    loss_train_avg = []
    
    start_time = time.time()

    #store shape of tensor per layer (for reshaping the numpy arrays)
    d = collections.OrderedDict()
    error = collections.OrderedDict()
    u = collections.OrderedDict()
    
    for i, ke in enumerate(glob_params):
        d[i] = glob_params[i].shape  
        u[i] = torch.ones(d[i]) * 1e-8
        error[i] = torch.zeros(d[i])
        
    local_u = [u]*args.num_users
    if args.error == 1:
        local_error = [error]*args.num_users
       
    if args.tau > 0:
        past_global_w = [glob_params]*args.tau
    
    glob = LocalAMSGrad(net_glob.parameters(), lr=args.lr, 
                                            betas=(args.momentum, args.beta2), 
                                            eps=1e-8, weight_decay=args.wdecay, amsgrad=True)
    
    iter_per_epoch = int(len(dataset_train) / args.bs)
    
    torch.manual_seed(0)
    low_acc_count = 0
    for epoch in range(args.epochs):
        if args.dataset == 'cifar':
            if epoch>=int(0.4*args.epochs) or epoch>=int(0.8*args.epochs):
                glob.param_groups[0]['lr'] = args.lr /10
        
        idxs_users = list(range(args.num_users))
        
        if args.customarg==1 and args.dataset == 'mnist':
            batch_dict = mnist_iid(dataset_train, iter_per_epoch)       
        if args.customarg==1 and args.dataset == 'cifar':
            batch_dict = cifar_iid(dataset_train, iter_per_epoch)
        if args.customarg==1 and args.dataset == 'imdb':
            batch_dict = cifar_iid(dataset_train, iter_per_epoch)    
        if args.customarg==1 and args.dataset == 'tinyimagenet':
            batch_dict = cifar_iid(dataset_train, iter_per_epoch)    
            
        for iter in range(iter_per_epoch):
            print('Starting Epoch {:3d}'.format(epoch)+ ' iteration {:3d}'.format(iter))
            batch_idx = list(batch_dict[iter])
            dict_users = mnist_iid(batch_idx, args.num_users)
            
            grad_locals, loss_locals = [], []
            for idx in idxs_users:
#                print('Starting for worker nb {:3d}'.format(idx))
                idx_set = [batch_idx[i] for i in dict_users[idx]]
                local = LocalUpdateAMS(args=args, dataset=dataset_train, idxs=set(idx_set))
                
                if args.tau > 0:
                    if epoch == 0 and iter < args.tau:
                        past_global_w[iter] = glob_params
                        gg = local.compute_gradient(net=copy.deepcopy(net_glob).to(args.device))
                    else:
                        stale_net_glob.load_state_dict(past_global_w[0])
                        gg = local.compute_gradient(net=copy.deepcopy(stale_net_glob).to(args.device))
                else:          
                    gg = local.compute_gradient(net=copy.deepcopy(net_glob).to(args.device))
                
                if args.method == 'full':
                    compress_grad = gg
                
                if args.method == 'unif':
                    compress_grad = collections.OrderedDict()
                    for i, ke in enumerate(glob_params):
                        compress_grad[i] = sparse.compute_g_tilde_unif(gg[i],args.sparsity).float()
                    
                if args.method == 'tong':
                    compress_grad = collections.OrderedDict()
                    for i, ke in enumerate(glob_params):
                        if args.error == 0:
                            prob = sparse.compute_prob_Tong(gg[i],args.sparsity)
                            compress_grad[i] = sparse.compute_g_tilde(gg[i],prob).float()
                        else:
                            prob = sparse.compute_prob_Tong(gg[i]+local_error[idx][i],args.sparsity)
                            compress_grad[i] = sparse.compute_g_tilde(gg[i]+local_error[idx][i],prob).float()
                            local_error[idx][i] = local_error[idx][i] + gg[i] - compress_grad[i]
                    
                if args.method == 'topk':
                    compress_grad = collections.OrderedDict()
                    for i, ke in enumerate(glob_params):
                        if args.error == 0:
                            compress_grad[i] = sparse.compute_topk(gg[i],max(args.sparsity,1/gg[i].numel())).float()
                        else:
                            compress_grad[i] = sparse.compute_topk(gg[i]+local_error[idx][i],max(args.sparsity,1/gg[i].numel())).float()
                            local_error[idx][i] = local_error[idx][i] + gg[i] - compress_grad[i]
                    
                if args.method == 'sign':
                    compress_grad = collections.OrderedDict()
                    for i, ke in enumerate(glob_params):
                        if args.error == 0:
                            compress_grad[i] = sparse.sign_grad(gg[i]).float()
                        else:
                            compress_grad[i] = sparse.sign_grad(gg[i]+local_error[idx][i]).float()
                            local_error[idx][i] = local_error[idx][i] + gg[i] - compress_grad[i]
                
                
                if args.tau == 0:
                    grad_locals.append(compress_grad)
                else:
                    glob.step(gg=compress_grad)
                    glob_params = net_glob.state_dict()
                    
                    del past_global_w[0]
                    past_global_w.append(glob_params)
                    
            if args.tau == 0:
                ave_grad = FedAvg(grad_locals)
                glob.step(gg=ave_grad)
                for i,p in enumerate(net_glob.parameters()):
                    glob_params[i] = p
                    

#        # print loss
#        loss_avg = sum(loss_locals) / len(loss_locals)
#        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
#        loss_train_avg.append(loss_avg)

        # testing
        net_glob.eval()
        acc_train, loss_train = test_fct(net_glob, dataset_train, args)
        acc_test, loss_test = test_fct(net_glob, dataset_test, args)
        print('Train loss {:.3f}'.format(loss_train))
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))

        elapsed_time = time.time() - start_time
        logger.append([args.lr, loss_train, acc_train, loss_test, acc_test, elapsed_time])
        
        if acc_test<20:
            low_acc_count+=1
        if low_acc_count==10:
            break
