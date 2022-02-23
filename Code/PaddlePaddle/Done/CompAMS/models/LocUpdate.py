#!/usr/bin/env python
import warnings
warnings.simplefilter("ignore", UserWarning)

import paddle
from paddle.io import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from .localams2 import LocalAMSGrad
import collections
#from .localsgd import LocalSGD

import pdb

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


#class LocalUpdate(object):
#    def __init__(self, args, dataset=None, idxs=None):
#        self.args = args
#        self.loss_func = nn.CrossEntropyLoss()
#        self.selected_clients = []
#        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
#
#    def train(self, net):
#        net.train()
#        # train and update
#        optimizer = LocalSGD(net.parameters(), lr=self.args.lr, momentum=0, LAMB=self.args.LAMB, 
#                             lambda0=self.args.lambda0)
#        
#        epoch_loss = []
#        for iter in range(self.args.local_ep):
#            batch_loss = []
#            for batch_idx, (images, labels) in enumerate(self.ldr_train):
#                images, labels = images.to(self.args.device), labels.to(self.args.device)
#                net.zero_grad()
#                log_probs = net(images)
#                loss = self.loss_func(log_probs, labels)
#                loss.backward()
#                optimizer.step()
#                if self.args.verbose and batch_idx % 10 == 0:
#                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
#                               100. * batch_idx / len(self.ldr_train), loss.item()))
#                batch_loss.append(loss.item())
#            epoch_loss.append(sum(batch_loss)/len(batch_loss))
#        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateAMS(object):
    def __init__(self, args, dataset=None, idxs=None, num_round=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
#        self.num_round = num_round

    def train(self, net , grad):
        net.train()
        # train and update with AMSGrad optimizer
        optimizer = LocalAMSGrad(net.parameters(), lr=self.args.lr, 
                                            betas=(self.args.momentum, self.args.beta2), 
                                            eps=1e-8, weight_decay=self.args.wdecay, amsgrad=True, 
                                            gg=grad)
        
        if grad is not None:
            optimizer.step()
        else:
            epoch_loss = []
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    if self.args.verbose and batch_idx % 10 == 0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                   100. * batch_idx / len(self.ldr_train), loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
#            ll = sum(epoch_loss) / len(epoch_loss)
        return net.state_dict() 

    def compute_gradient(self, net):
        net.train()
        # simply returning the gradient of a network
#        optimizer = LocalAMSGrad(net.parameters(), lr=self.args.lr, 
#                                            betas=(self.args.momentum, self.args.beta2), 
#                                            eps=1e-8, weight_decay=self.args.wdecay, amsgrad=True, 
#                                            v_hat=v_hat, v=v, m=m, num_round=self.num_round, LAMB=self.args.LAMB,
#                                            lambda0=self.args.lambda0)

        gg=collections.OrderedDict()
        i=0
        for p in net.parameters():
            gg[i]=paddle.zeros_like(p)
            i+=1
            
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
            i=0
            for p in net.parameters():
                gg[i]+=p.grad/len(self.ldr_train)
                i+=1
                
        return gg